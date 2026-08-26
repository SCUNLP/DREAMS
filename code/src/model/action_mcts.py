"""Action-level Monte Carlo tree search for DREAMS."""

import copy
import json
import math
import random
import time

import numpy as np
from loguru import logger
from tqdm import tqdm

from .early_policy import select_early_inquiry
from .errors import ActionExecutionError, StateError
from .llm_client import CHAT_MODEL, call_chatgpt_1
from .parallel import iter_completed


GPT4 = CHAT_MODEL


class MCTSNode:
    def __init__(self, crs_agent, state, parent=None, action=None, trace=None, transition=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = self.get_possible_actions()  # Actions not yet explored
        self.crs_agent = crs_agent
        self.trace = trace
        self.transition = transition

    def get_possible_actions(self):
        """Return a list of possible actions from this state."""
        # All possible actions in the CRS domain
        actions = [
            'GenreInquiry',
            'ActorInquiry',
            'DirectorInquiry',
            'ItemRecommendation',
            'ItemExplanation',
            'FailureReflection'
        ]

        return actions

    def select_child(self, exploration_weight=1.0):
        """Select a child node using UCB1 formula."""
        try:
            if not self.children:
                raise StateError("No children available for selection")

            # Prevent numerical issues in UCB1 calculation
            log_visits = math.log(max(self.visits, 1))  # Prevent log(0)

            def ucb(child):
                if child.visits == 0:
                    return float('inf')  # Prioritize unvisited nodes

                # Add numerical stability checks
                exploitation = child.value / max(child.visits, 1)  # Prevent division by zero
                exploration = exploration_weight * math.sqrt(2 * log_visits / max(child.visits, 1))

                exploration = min(exploration, 1000.0)

                return exploitation + exploration

            max_ucb = float('-inf')
            best_children = []

            for child in self.children.values():
                ucb_value = ucb(child)
                if ucb_value > max_ucb:
                    max_ucb = ucb_value
                    best_children = [child]
                elif ucb_value == max_ucb:
                    best_children.append(child)

            return random.choice(best_children)

        except Exception as e:
            logger.error(f"Error in select_child: {str(e)}")
            raise ActionExecutionError(f"Failed to select child: {str(e)}")

    def expand(self, predicted_action_scores):
        """Expand the node by adding a child node for an untried action."""
        try:
            if not self.untried_actions:
                logger.warning("No untried actions available for expansion")
                return None
            valid_actions_with_scores = [
                (action, predicted_action_scores.get(action, 0))
                for action in self.untried_actions
                if predicted_action_scores.get(action, 0) > 0]
            if valid_actions_with_scores:
                best_action = max(valid_actions_with_scores, key=lambda x: x[1])[0]
                state = copy.deepcopy(self.state)
                if not self.transition:
                    raise StateError("MCTS node is missing its state transition")
                next_state = self.transition(state, best_action)
                child = MCTSNode(
                    self.crs_agent,
                    next_state,
                    parent=self,
                    action=best_action,
                    trace=self.trace,
                    transition=self.transition,
                )
                self.children[best_action] = child
                self.untried_actions.remove(best_action)
                return child

            else:
                logger.warning("No actions with positive scores found")
            return None

        except Exception as e:
            logger.error(f"Error in expand: {str(e)}")
            raise ActionExecutionError(f"Failed to expand node: {str(e)}")

    def update(self, reward):
        """Update the node statistics with the reward."""
        try:
            if not isinstance(reward, (int, float)):
                raise ValueError("Reward must be a numeric value")
            self.visits += 1
            self.value += reward
        except Exception as e:
            logger.error(f"Error in update: {str(e)}")
            raise ActionExecutionError(f"Failed to update node: {str(e)}")

    def is_fully_expanded(self):
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """Check if the node is terminal (e.g., conversation ended)."""
        return self.state.get('user_attitude') == 'accepted' or self.state.get('turn_count', 0) > 10


class MCTS:
    def __init__(
        self,
        crs_agent,
        simulation_count=3,
        exploration_weight=1.0,
        early_inquiry_rounds=2,
        simulation_workers=1,
        trace=None,
    ):
        self.crs_agent = crs_agent
        self.simulation_count = max(1, simulation_count)
        self.exploration_weight = exploration_weight
        self.early_inquiry_rounds = early_inquiry_rounds
        self.simulation_workers = max(1, simulation_workers)
        self.trace = trace
        self.virtual_loss = 1.0

    def search(self, state):
        """Run the MCTS algorithm and return the best action."""
        predicted_action_scores, reason = self.get_llm_action_policy(state)
        if self.trace:
            self.trace.record("policy", scores=predicted_action_scores, reason=reason)

        # Deliberate efficiency policy: the first two real turns ask for
        # preferences without rollout/backpropagation.
        if len(state['actions_taken']) < self.early_inquiry_rounds:
            action = select_early_inquiry(predicted_action_scores, state['actions_taken'])
            if self.trace:
                self.trace.record("fast_inquiry", action=action, rollout=False)
                self.trace.render(state=state)
            return action

        root = MCTSNode(self.crs_agent, state, trace=self.trace, transition=self._expand_action)
        if self.simulation_workers == 1:
            self._run_sequential(root, predicted_action_scores, state)
        else:
            self._run_parallel(root, predicted_action_scores, state)

        action = self.best_action(root, predicted_action_scores)
        if self.trace:
            self.trace.record("selected_action", action=action)
            self.trace.render(root, state)
        return action

    def _expand_action(self, state, action):
        agent = self.crs_agent.clone_for_simulation()
        return self.apply_action(state, action, agent, "expansion")

    def _run_sequential(self, root, predicted_action_scores, state):
        for simulation_id in tqdm(range(self.simulation_count), desc="MCTS Search", leave=False):
            node = self.select(root, predicted_action_scores)
            if self.trace:
                self.trace.record("tree_expanded", simulation=simulation_id, action=node.action)
                self.trace.render(root, state)
            reward = self.simulate(node, simulation_id)
            self.backpropagate(node, reward)
            if self.trace:
                self.trace.record("backpropagation", simulation=simulation_id, action=node.action, reward=reward)
                self.trace.render(root, state)

    def _run_parallel(self, root, predicted_action_scores, state):
        # Batched MCTS: selection/expansion stays serial; independent rollout
        # simulations run concurrently and are merged through serial backprop.
        selected_nodes = []
        for simulation_id in range(self.simulation_count):
            node = self.select(root, predicted_action_scores)
            self._apply_virtual_loss(node)
            selected_nodes.append((simulation_id, node))
            if self.trace:
                self.trace.record("tree_expanded", simulation=simulation_id, action=node.action)
                self.trace.render(root, state)

        worker_count = min(self.simulation_workers, len(selected_nodes))

        def rollout(job):
            simulation_id, node = job
            try:
                return self.simulate(node, simulation_id)
            except Exception as error:
                return error

        for (simulation_id, node), reward in iter_completed(rollout, selected_nodes, worker_count):
            self._remove_virtual_loss(node)
            if isinstance(reward, Exception):
                if self.trace:
                    self.trace.record("simulation_error", simulation=simulation_id, error=str(reward))
                raise reward
            self.backpropagate(node, reward)
            if self.trace:
                self.trace.record("backpropagation", simulation=simulation_id, action=node.action, reward=reward)
                self.trace.render(root, state)

    def _apply_virtual_loss(self, node):
        current = node
        while current is not None:
            current.visits += 1
            current.value -= self.virtual_loss
            current = current.parent

    def _remove_virtual_loss(self, node):
        current = node
        while current is not None:
            current.visits -= 1
            current.value += self.virtual_loss
            current = current.parent

    def get_llm_action_policy(self, state):
        """Use LLM to predict which actions would be most appropriate for the current state."""
        state_description = self.format_state_for_llm(state)

        prompt = [
            {"role": "system", "content":
            """You are an AI assistant helping to determine the best action for a conversational movie recommender system.
            Based on the current conversation state(mainly focusing on the user preference, user attitude, and recent actions), rank the following actions from most to least appropriate:
            - GenreInquiry: Ask about the user's preferred movie genres
            - ActorInquiry: Ask about the user's preferred actors
            - DirectorInquiry: Ask about the user's preferred directors
            - ItemRecommendation: When explicitly knowing all of the user preference(genre, actor, director), retrieve for the movie item and recommend it to the user.
            - ItemExplanation: After you recommend a movie, if the user is interested in the recommended movie and ask for details, provide details about the recommended movie to the user
            - FailureReflection: When your recommendation is rejected by the user, reflect on the fundamental reasons for being rejected and take corresponding measures.


            Here are some empirical rules:
            1.	When the attribute of user preference in the state is empty, start with GenreInquiry, then narrowing down with ActorInquiry or DirectorInquiry.
            2.  Attention! Unless the user has already expressed their preference about all the attributes(genres, actors, directors), you should consider the user preference as not clear.
	        3.  After you get the explicit user preference about all the attributes(genres, actors, directors), you should conduct ItemRecommendation.
	        4.	Use FailureReflection for recovery: If recommendations fail(user attitude is rejected), reset by revisiting preference inquiries.
	        5.	Enhance recommendations with explanations: After Recommendation, provide ItemExplanation to improve user trust.
            6.  Do not proceed with ItemExplanation before ItemRecommendation has been conducted.
            7.  If the user continuously ask about the details of the movie in successive rounds, you can continuously adopt "ItemExplanation" strategie.

            Here are some examples of optimal decision paths:
            Example 1:
            GenreInquiry -> ActorInquiry -> DirectorInquiry -> ItemRecommendation -> ItemExplanation
            Example 2:
            GenreInquiry -> ActorInquiry -> ItemRecommendation -> ItemExplanation -> FailureReflection -> DirectorInquiry -> ItemRecommendation -> ItemExplanation


            Also, you should give your reason for the score of each action.
            Return a JSON object with each action and a score from 0-10 where 10 is most appropriate, and a reason for the score:
            {
                "GenreInquiry": score,
                "ActorInquiry": score,
                "DirectorInquiry": score,
                "ItemRecommendation": score,
                "ItemExplanation": score,
                "FailureReflection": score,
                "reason": reason
            }
            """},
            {"role": "user", "content": f"Current state of the conversation:\n{state_description}\n\nRank the actions for this state."}
        ]

        try:
            action_scores_json = call_chatgpt_1(prompt, model_name=GPT4, temperature=0.2, seed=0, json_mode=True)
            action_scores = json.loads(action_scores_json)
            reason = action_scores.pop('reason', None)

            total_score = sum(action_scores.values())
            if total_score > 0:
                for action in action_scores:
                    action_scores[action] = action_scores[action] / total_score


            return action_scores, reason

        except Exception as e:
            logger.error(f"Error getting LLM action policy: {e}")
            actions = [
                "GenreInquiry", "ActorInquiry", "DirectorInquiry", "ItemRecommendation", "ItemExplanation", "FailureReflection"
            ]
            return {action: 1/len(actions) for action in actions}, "uniform fallback after policy error"

    def format_state_for_llm(self, state):
        """Format the state into a readable description for the LLM."""
        description = "CONVERSATION STATE:\n"

        # User preferences
        description += "User preferences:\n"
        if state['user_preferences']['genres']:
            description += f"- Genres: {', '.join(state['user_preferences']['genres'])}\n"
        else:
            description += "- Genres: None expressed\n"

        if state['user_preferences']['actors']:
            description += f"- Actors: {', '.join(state['user_preferences']['actors'])}\n"
        else:
            description += "- Actors: None expressed\n"

        if state['user_preferences']['directors']:
            description += f"- Directors: {', '.join(state['user_preferences']['directors'])}\n"
        else:
            description += "- Directors: None expressed\n"

        # Recommended items
        description += "\nRecommended items:\n"
        if state['recommended_items']:
            for i, item in enumerate(state['recommended_items']):
                description += f"- {item}\n"
        else:
            description += "- No recommendations made yet\n"

        # User attitude
        description += f"\nUser attitude: {state.get('user_attitude', 'undecided')}\n"

        # Turn count
        description += f"Turn count: {state.get('turn_count', 0)}\n"

        # Recent actions
        description += "\nRecent actions:\n"
        recent_actions = state['actions_taken'][-3:] if state['actions_taken'] else []
        if recent_actions:
            for action in recent_actions:
                description += f"- {action}\n"
        else:
            description += "- No actions taken yet\n"

        description += "\nRecent conversation:\n"
        recent_context = state['context'][-4:] if len(state['context']) >= 4 else state['context']
        for i, message in enumerate(recent_context):
            role = "User" if i % 2 == 0 else "Assistant"
            description += f"{role}: {message}\n"

        return description

    def select(self, node, predicted_action_scores):
        """Select a node to explore further."""
        try:
            max_iterations = 100  # Prevent infinite loops
            iterations = 0
            while not node.is_terminal() and iterations < max_iterations:
                if not node.is_fully_expanded():
                    expanded_node = node.expand(predicted_action_scores)
                    if expanded_node is None:
                        # If expansion failed, try to select a child if available
                        if node.children:
                            node = node.select_child(self.exploration_weight)
                        else:
                            # If no children available, return the current node
                            return node
                    else:
                        return expanded_node
                else:
                    node = node.select_child(self.exploration_weight)
                iterations += 1

            if iterations >= max_iterations:
                logger.warning("Maximum iterations reached in select method")

            return node
        except Exception as e:
            logger.error(f"Error in select: {str(e)}")
            raise ActionExecutionError(f"Failed to select node: {str(e)}")

    def simulate(self, node, simulation_id=None):
        """Simulate a random playout from the given node, prioritizing user acceptance."""
        try:
            state = copy.deepcopy(node.state)
            simulation_agent = self.crs_agent.clone_for_simulation()
            seed = (self.crs_agent.seed or 0) + (simulation_id or 0) + len(state.get('actions_taken', []))
            rng = np.random.default_rng(seed)
            depth = 0
            max_depth = 3  # Limit the simulation depth
            cumulative_reward = 0
            discount_factor = 0.95  # Increased from 0.9 to give more weight to future rewards

            acceptance_achieved = False
            acceptance_depth = -1

            start_time = time.time()
            max_simulation_time = 50  # Maximum time in seconds for simulation
            if self.trace:
                self.trace.record("simulation_started", simulation=simulation_id, action=node.action)

            while depth < max_depth and not self.is_terminal_state(state):
                # Check for timeout
                if time.time() - start_time > max_simulation_time:
                    logger.warning(f"Simulation timed out after {max_simulation_time} seconds")
                    if self.trace:
                        self.trace.record("simulation_timeout", simulation=simulation_id, seconds=max_simulation_time)
                    break

                try:
                    action_scores, reason = self.get_llm_action_policy(state)
                    if not action_scores:
                        logger.warning("No action scores returned from LLM")
                        break

                    actions = list(action_scores.keys())
                    scores = np.array([action_scores[a] for a in actions])


                    if (len(state['user_preferences']['genres']) >= 1 or
                        len(state['user_preferences']['actors']) >= 1 or
                        len(state['user_preferences']['directors']) >= 1):
                        for i, action in enumerate(actions):

                            if action in ["ItemRecommendation", "ItemExplanation"]:
                                scores[i] *= 1.2

                    noise_param = max(0.1, 0.3 - (depth * 0.05))  # Reduce noise with depth
                    exploration_noise = rng.dirichlet([noise_param] * len(scores))
                    scores = 0.9 * scores + 0.1 * exploration_noise

                    # Normalize scores
                    total_score = np.sum(scores)
                    if total_score == 0:
                        scores = np.ones_like(scores) / len(scores)
                    else:
                        scores = scores / total_score

                    action = rng.choice(actions, p=scores)

                    state = self.apply_action(state, action, simulation_agent, simulation_id)

                    immediate_reward = self.evaluate_state(state)

                    # Check if user has accepted a recommendation
                    if state.get('user_attitude') == 'accepted' and not acceptance_achieved:
                        acceptance_achieved = True
                        acceptance_depth = depth
                        # Give a bonus for achieving acceptance
                        immediate_reward += 10.0

                    cumulative_reward += (discount_factor ** depth) * immediate_reward
                    if self.trace:
                        self.trace.record(
                            "reward",
                            simulation=simulation_id,
                            depth=depth,
                            action=action,
                            immediate=immediate_reward,
                            cumulative=cumulative_reward,
                            attitude=state.get('user_attitude'),
                        )

                    if len(state.get('actions_taken', [])) > 20 or len(state.get('context', [])) > 20:
                        logger.warning("State size exceeded maximum allowed")
                        break

                    depth += 1

                except Exception as e:
                    logger.error(f"Error during simulation iteration: {str(e)}")
                    break


            if self.is_terminal_state(state):
                final_reward = self.evaluate_state(state)

                if state.get('user_attitude') == 'accepted':
                    final_reward *= 1.5

                cumulative_reward += (discount_factor ** depth) * final_reward


            if acceptance_achieved:
                early_acceptance_bonus = 3.0 * (1.0 - (acceptance_depth / max_depth))
                cumulative_reward += early_acceptance_bonus
            elif depth >= max_depth and state.get('user_attitude') == 'undecided':
                # Small penalty for reaching max depth without decision
                cumulative_reward -= 1.0

            if self.trace:
                self.trace.record(
                    "simulation_finished",
                    simulation=simulation_id,
                    reward=cumulative_reward,
                    seconds=round(time.time() - start_time, 3),
                )
            return cumulative_reward

        except Exception as e:
            logger.error(f"Error in simulate: {str(e)}")
            raise ActionExecutionError(f"Failed to simulate playout: {str(e)}")

    def backpropagate(self, node, reward):
        """Update statistics for all nodes on the path from the node to the root."""
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self, node, predicted_scores=None):
        """Return the best action based on visit count and LLM predictions."""
        try:
            if not node.children:
                if predicted_scores:
                    return max(predicted_scores.items(), key=lambda x: x[1])[0]
                else:
                    return self.random_action(node.state)

            combined_scores = {}

            total_visits = sum(child.visits for child in node.children.values())
            if total_visits == 0:
                if predicted_scores:
                    return max(predicted_scores.items(), key=lambda x: x[1])[0]
                else:
                    return self.random_action(node.state)

            for action, child in node.children.items():
                mcts_score = child.visits / total_visits

                llm_score = predicted_scores.get(action, 0) if predicted_scores else 0
                combined_scores[action] = (0.5 * mcts_score) + (0.5 * llm_score)
            return max(combined_scores.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"Error in best_action: {str(e)}")
            raise ActionExecutionError(f"Failed to determine best action: {str(e)}")

    def random_action(self, state):
        """Select a random action for the simulation."""
        possible_actions = [
            'GenreInquiry',
            'ActorInquiry',
            'DirectorInquiry',
            'ItemRecommendation',
            'ItemExplanation',
            'FailureReflection'
        ]
        return random.choice(possible_actions)

    def apply_action(self, state, action, simulation_agent=None, simulation_id=None):
        """Apply the given action to the state and return the new state."""
        new_state = copy.deepcopy(state)
        simulation_agent = simulation_agent or self.crs_agent.clone_for_simulation()

        new_state['actions_taken'].append(action)
        new_state['turn_count'] = new_state.get('turn_count', 0) + 0.5  # Increment by 0.5 since a full turn is user+assistant

        try:
            conv_dict = {
                'context': new_state.get('context', []),
                'rec': new_state.get('recommended_items', [])
            }

            result = simulation_agent.execute_action(action, conv_dict, simulate=True)
            if len(result) == 2:
                meta_info, assistant_response = result
            elif len(result) == 4:
                cand_list, meta_info, assistant_response, refined_query = result

            new_state['context'].append(assistant_response)

            user_feedback = self._simulate_user_feedback(action, new_state, meta_info)
            new_state['context'].append(user_feedback)
            self._update_state_from_feedback(new_state, user_feedback)
            if self.trace:
                self.trace.record(
                    "feedback",
                    simulation=simulation_id,
                    action=action,
                    assistant=assistant_response,
                    feedback=user_feedback,
                )

        except Exception as e:
            logger.error(f"Error simulating action {action}: {str(e)}")
            new_state['context'].append(f"Assistant: I'm taking action {action}")
            new_state['context'].append("User: I understand")

        return new_state

    def _simulate_user_feedback(self, action, state, meta_info):
        """Simulate user feedback based on the action and current state."""
        try:
            feedback_prompt = [
                {"role": "system", "content": f"""You are simulating a user's response in a movie recommendation conversation.
                Based on the assistant's action and response, generate a realistic user response.
                Consider the following:
                1. The user's current preferences and attitude
                2. The type of action the assistant just took
                3. The natural flow of conversation

                Your preferred movie should cover those genres at the same time: {state['user_preferences']['genres']}.
                Your preferred movie should cover at least one of these stars: {state['user_preferences']['actors']}.
                Your preferred movie should cover at least one of these directors: {state['user_preferences']['directors']}.
                You must follow the instructions below during chat.
                1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie. However, do not continuously ask for particularly detailed content like "memorable scenes".
                2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
                3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
                4. Once you are sure that the recommended movie exactly covers all your preferred genres, and covers at least one of the actors and directors you like, you should accept it and end the conversation with a special token "[END]" at the end of your response.
                5. If the recommender asks your preferred genre, you should describe your preferred movie genre in your own words and you'd better not clearly mention the movie type.
                6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent.
                7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!
                8. If the recommender asks you about your preferred genre, actors, or directors, you should always answer it.
                9. In a single conversation, only reveal one of your preferred genres, actors, or directors.
                10. If the recommender tells you that the star and director you like have never collaborated, try telling them about other stars or directors you like.

                Keep the response brief and realistic."""},
                {"role": "user", "content": f"""Current state:
                - User preferences: {state['user_preferences']}
                - User attitude: {state.get('user_attitude', 'undecided')}
                - Recent actions: {state['actions_taken'][-3:]}
                - Assistant's last response: {state['context'][-1] if state['context'] else 'None'}

                Assistant's action: {action}
                Meta info: {meta_info}

                Generate a realistic user response."""}
            ]

            # Get simulated user feedback from LLM
            user_feedback = call_chatgpt_1(feedback_prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

            return user_feedback

        except Exception as e:
            logger.error(f"Error simulating user feedback: {str(e)}")
            return "User: I understand"

    def _update_state_from_feedback(self, state, user_feedback):
        """Update the state based on the simulated user feedback."""
        try:
            extraction_prompt = [
                {"role": "system", "content": """Analyze the user's response and extract:
                1. Any new preferences (genres, actors, directors)
                2. User's attitude (accepted, rejected, curious, undecided)
                3. Confidence in attitude assessment (0-1)

                Return a JSON object with this information."""},
                {"role": "user", "content": f"User response: {user_feedback}"}
            ]

            analysis_json = call_chatgpt_1(extraction_prompt, model_name=GPT4, temperature=0.1, seed=0, json_mode=True)
            analysis = json.loads(analysis_json)

            # Update preferences
            preferences = analysis.get('preferences', {})
            for genre in preferences.get('genres', []):
                if genre.lower() not in [g.lower() for g in state['user_preferences']['genres']]:
                    state['user_preferences']['genres'].append(genre.lower())

            for actor in preferences.get('actors', []):
                if actor.lower() not in [a.lower() for a in state['user_preferences']['actors']]:
                    state['user_preferences']['actors'].append(actor.lower())

            for director in preferences.get('directors', []):
                if director.lower() not in [d.lower() for d in state['user_preferences']['directors']]:
                    state['user_preferences']['directors'].append(director.lower())

            # Update user attitude if confidence is high enough
            attitude = analysis.get('attitude', 'undecided')
            confidence = analysis.get('confidence', 0.0)

            if confidence > 0.4:  # Only update if confidence is reasonable
                state['user_attitude'] = attitude

        except Exception as e:
            logger.error(f"Error updating state from feedback: {str(e)}")

    def is_terminal_state(self, state):
        """Check if the state is terminal based on user attitude or turn count."""
        return state.get('user_attitude') == 'accepted' or state.get('turn_count', 0) > 10

    def evaluate_state(self, state):
        """Evaluate the state and return a reward that prioritizes user attitude."""
        try:
            if state.get('user_attitude') == 'accepted':
                return 10.0

            if state.get('user_attitude') == 'curious':
                return 5.0

            if state.get('user_attitude') == 'rejected':
                return -2.0

            info_gathered = 0
            if state['user_preferences']['genres']:
                info_gathered += len(state['user_preferences']['genres']) * 0.1
            if state['user_preferences']['actors']:
                info_gathered += len(state['user_preferences']['actors']) * 0.1
            if state['user_preferences']['directors']:
                info_gathered += len(state['user_preferences']['directors']) * 0.1

            info_reward = min(0.5, info_gathered)

            action_sequence = state.get('actions_taken', [])
            has_inquiry = any(action in action_sequence for action in ["GenreInquiry", "ActorInquiry", "DirectorInquiry"])
            has_retrieval = "ItemRecommendation" in action_sequence or "get_rec" in action_sequence
            has_reranking = "Reranking" in action_sequence
            has_explanation = "ItemExplanation" in action_sequence

            trajectory_score = 0
            if has_inquiry:
                trajectory_score += 0.5
            if has_retrieval and has_inquiry:
                trajectory_score += 0.8
            if has_explanation and has_reranking:
                trajectory_score += 1.0

            recommendation_score = 0
            if state.get('recommended_items'):
                recommendation_score += 0.5

                if info_gathered > 0.3:
                    recommendation_score += 1.0

            turn_penalty = max(0, (state.get('turn_count', 0) - 5) * 0.1)

            total_reward = info_reward + trajectory_score + recommendation_score - turn_penalty

            total_reward = min(total_reward, 5.0)

            return total_reward
        except Exception as e:
            logger.error(f"Error in evaluate_state: {str(e)}")
            raise ActionExecutionError(f"Failed to evaluate state: {str(e)}")
