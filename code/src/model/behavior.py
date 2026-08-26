"""Conversation actions, retrieval, reranking, and response generation."""

import copy
import difflib
import json
import math
import random
from collections import Counter

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from .errors import ActionExecutionError, LLMError, StateError
from .llm_client import (
    CHAT_MODEL,
    call_chatgpt_1,
    call_chatgpt_2,
    call_chatgpt_3,
    call_embedding,
)


GPT4 = CHAT_MODEL


class AgentBehaviorMixin:
    def execute_action(self, action, conv_dict, simulate=False):
        """Execute the selected action and update state."""
        try:
            if not isinstance(conv_dict, dict) or 'context' not in conv_dict:
                raise StateError("Invalid conversation dictionary format")

            if not simulate:
                self.conversation_state['actions_taken'].append(action)

            if action == "GenreInquiry":
                return self.genre_inquiry(conv_dict)
            elif action == "ActorInquiry":
                return self.actor_inquiry(conv_dict)
            elif action == "DirectorInquiry":
                return self.director_inquiry(conv_dict)
            elif action == "ItemRecommendation":
                conv_paraphrase, item_rank_arr = self.conv_paraphrase(conv_dict)
                a = {}
                b = ""
                c = conv_paraphrase.get('refined_query')
                a, b = self.reranking(conv_paraphrase)
                if not simulate:
                    return item_rank_arr, a, b, c
                else:
                    return a, b
            elif action == "ItemExplanation":
                if not simulate and not self.movie_info.get("name"):
                    return self.get_conv(conv_dict)
                elif simulate and not self.movie_info_sim.get("name"):
                    return self.get_conv(conv_dict)
                else:
                    return self.item_explanation(conv_dict, simulate)
            elif action == "FailureReflection":
                return self.failure_reflection(conv_dict, simulate)
            else:
                return self.get_conv(conv_dict)
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}")
            raise ActionExecutionError(f"Failed to execute action {action}: {str(e)}")

    def genre_inquiry(self, conv_dict):
        """Ask about the user's preference for movie genres."""
        try:
            prompt = self.construct_inquiry_prompt(conv_dict, "genre")
            meta_info = {}

            gen_str = call_chatgpt_2(prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

            meta_info['ans_type'] = 'inquiry'
            meta_info['inquiry_type'] = 'genre'

            return meta_info, gen_str
        except Exception as e:
            logger.error(f"Error in genre inquiry: {str(e)}")
            raise ActionExecutionError(f"Failed to execute genre inquiry: {str(e)}")

    def actor_inquiry(self, conv_dict):
        """Ask about the user's preference for actors."""
        try:
            prompt = self.construct_inquiry_prompt(conv_dict, "actor")
            meta_info = {}

            gen_str = call_chatgpt_2(prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

            meta_info['ans_type'] = 'inquiry'
            meta_info['inquiry_type'] = 'actor'

            return meta_info, gen_str
        except Exception as e:
            logger.error(f"Error in actor inquiry: {str(e)}")
            raise ActionExecutionError(f"Failed to execute actor inquiry: {str(e)}")

    def director_inquiry(self, conv_dict):
        """Ask about the user's preference for directors."""
        try:
            prompt = self.construct_inquiry_prompt(conv_dict, "director")
            meta_info = {}

            gen_str = call_chatgpt_2(prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

            meta_info['ans_type'] = 'inquiry'
            meta_info['inquiry_type'] = 'director'

            return meta_info, gen_str
        except Exception as e:
            logger.error(f"Error in director inquiry: {str(e)}")
            raise ActionExecutionError(f"Failed to execute director inquiry: {str(e)}")

    def conv_paraphrase(self, conv_dict):
        """Refine the query for the retrieval model by paraphrasing extracted user preferences
        using MCTS to optimize the paraphrasing strategy."""
        try:
            context = conv_dict['context']
            genre_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                      'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
                      'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                      'war', 'western']

            extraction_prompt = [
                {"role": "system", "content": """You are an information extraction expert.
                    Extract the user's movie preferences from the conversation.
                    Focus on extracting specific:
                    1. Genres they like/dislike
                    2. Actors they like/dislike
                    3. Directors they like/dislike
                    4. Plot elements or themes they enjoy
                    5. Your confidence in information extraction

                    Genres should be from the following list: {'action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                    'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                    'war', 'western'}
                    Here are some descriptions of movie genres that are easy to confuse:
                        'action': 'thrilling and adrenaline-pumping action movie'
                        'crime': 'suspenseful and intense criminal film'
                        'mystery': 'intriguing and suspenseful mystery',
                        'thriller': 'suspenseful and thrilling thriller with gripping suspense'
                    The descriptions of these genres are very similar; you need to carefully distinguish which genre the user's description actually belongs to.
                    Return ONLY a JSON object with these preferences:
                    {
                        "genres": {"liked": [], "disliked": [], 'confidence': []},
                        "stars": {"liked": [], "disliked": [], 'confidence': []},
                        "directors": {"liked": [], "disliked": [], 'confidence': []},
                        "themes": {"liked": [], "disliked": []},
                    }"""}
            ]

            for i, text in enumerate(context):
                if not isinstance(text, str):
                    text = str(text)
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                extraction_prompt.append({
                    'role': role_str,
                    'content': text
                })

            try:
                extracted_preferences_json = call_chatgpt_1(extraction_prompt, model_name=GPT4, temperature=0.1, seed=0, json_mode=True)
                extracted_preferences = json.loads(extracted_preferences_json)
            except Exception as ex:
                logger.error(f"Preference extraction failed: {str(ex)}")
                extracted_preferences = {
                    "genres": {"liked": [], "disliked": [], 'confidence': []},
                    "stars": {"liked": [], "disliked": [], 'confidence': []},
                    "directors": {"liked": [], "disliked": [], 'confidence': []},
                    "themes": {"liked": [], "disliked": [], 'confidence': []},
                }

                for genre in self.conversation_state['user_preferences']['genres']:
                    extracted_preferences["genres"]["liked"].append(genre)
                for actor in self.conversation_state['user_preferences']['actors']:
                    extracted_preferences["stars"]["liked"].append(actor)
                for director in self.conversation_state['user_preferences']['directors']:
                    extracted_preferences["directors"]["liked"].append(director)


            paraphrase_strategies = [
                "balanced",
                "genre_weighted",
                "person_weighted",
                "positive_only",
            ]

            class ParaphraseMCTSNode:
                def __init__(self, strategy=None, parent=None):
                    self.strategy = strategy
                    self.parent = parent
                    self.children = {}
                    self.visits = 0
                    self.value = 0

                def is_terminal(self):
                    return self.strategy is not None and len(self.children) == 0

                def expand(self, available_strategies):
                    for strategy in available_strategies:
                        if strategy not in self.children:
                            self.children[strategy] = ParaphraseMCTSNode(strategy=strategy, parent=self)
                    return list(self.children.values())

                def select_child(self, exploration_weight=1.0):
                    log_visits = math.log(max(self.visits, 1))
                    def ucb(child):
                        if child.visits == 0:
                            return float('inf')
                        exploitation = child.value / max(child.visits, 1)
                        exploration = exploration_weight * math.sqrt(2 * log_visits / max(child.visits, 1))
                        exploration = min(exploration, 1000.0)
                        return exploitation + exploration

                    return max(self.children.values(), key=ucb)

                def update(self, reward):
                    self.visits += 1
                    self.value += reward

            def generate_retrieval_query(strategy, preferences):
                confidence_map = {
                    "balanced": {
                        "genres": 1.0,
                        "stars": 1.0,
                        "directors": 1.0,
                        "themes": 1.0,
                        "use_negative": True
                    },
                    "genre_weighted": {
                        "genres": 2.0,
                        "stars": 0.7,
                        "directors": 0.7,
                        "themes": 1.0,
                        "use_negative": True
                    },
                    "person_weighted": {
                        "genres": 0.7,
                        "stars": 2.0,
                        "directors": 2.0,
                        "themes": 0.5,
                        "use_negative": True
                    },
                    "positive_only": {
                        "genres": 1.0,
                        "stars": 1.0,
                        "directors": 1.0,
                        "themes": 1.0,
                        "use_negative": False
                    },
                }

                # Build a query based on the weighted preferences
                query_parts = []
                weights = confidence_map[strategy]

                # Process each preference category
                for category in preferences:
                    if category in weights:
                        weight = weights[category]

                        # Add liked items
                        for item in preferences[category]["liked"]:
                            if item and len(item.strip()) > 0:
                                query_parts.append(item.strip())

                        # Add disliked items if strategy uses negatives
                        if weights.get("use_negative", False):
                            for item in preferences[category]["disliked"]:
                                if item and len(item.strip()) > 0:
                                    query_parts.append(f"NOT {item.strip()}")

                # Build the query template based on the strategy
                if len(query_parts) == 0:
                    # No specific preferences, create a generic query
                    query_template = "A movie with good ratings and engaging story"
                else:
                    query_template = " ".join(query_parts)

                # Final query construction with LLM
                query_prompt = [
                    {"role": "system", "content": f"""You are an expert at creating retrieval queries for movie recommendation systems.
                        I will give you raw preference data and your task is to create a well-formatted search query.

                        Current strategy: {strategy}
                        Raw preferences: {json.dumps(preferences)}

                        Create a concise, explicit and well-formated query that will perform well with a semantic retrieval model.
                        Your query should be structured for unleashing the best potential of the retriever.
                        After structuring the user preferences, you can shuffle the sequence of preferences for better retrieval.
                        Return ONLY the query text with no explanations."""}
                ]

                refined_query = call_chatgpt_1(query_prompt, model_name=GPT4, temperature=0.3, seed=0, json_mode=False)
                return refined_query

            # Evaluate a retrieval query using simulated retrieval
            def evaluate_query(query):
                try:
                    conv_embed = call_embedding(query).data[0].embedding
                    conv_embed = np.asarray(conv_embed).reshape(1, -1)

                    if len(self.item_emb_arr) == 0:
                        raise StateError("No item embeddings available")

                    sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
                    rank_arr = np.argsort(sim_mat, axis=-1).tolist()
                    rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
                    item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
                    item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

                    top_items = []
                    for i in range(min(5, len(item_rank_arr[0]))):  # Get more items for better evaluation
                        try:
                            top_items.append(self.id2info[self.entityid2id[item_rank_arr[0][i]]])
                        except (KeyError, IndexError):
                            continue

                    preference_score = 0

                    item_scores = []

                    for item in top_items:
                        item_score = 0

                        item_genres = item['genre'] if isinstance(item['genre'], list) else [item['genre']]
                        item_genres = [g.lower() for g in item_genres if isinstance(g, str)]

                        for genre in extracted_preferences["genres"]["liked"]:
                            if isinstance(genre, str) and genre.lower() in item_genres:
                                item_score += 1.0

                        for genre in extracted_preferences["genres"]["disliked"]:
                            if isinstance(genre, str) and genre.lower() in item_genres:
                                item_score -= 0.5

                        item_actors = item['star'] if isinstance(item['star'], list) else [item['star']]
                        item_actors = [a.lower() for a in item_actors if isinstance(a, str)]

                        for actor in extracted_preferences["stars"]["liked"]:
                            if isinstance(actor, str) and actor.lower() in item_actors:
                                item_score += 1.0

                        for actor in extracted_preferences["stars"]["disliked"]:
                            if isinstance(actor, str) and actor.lower() in item_actors:
                                item_score -= 0.5

                        item_directors = item['director'] if isinstance(item['director'], list) else [item['director']]
                        item_directors = [d.lower() for d in item_directors if isinstance(d, str)]

                        for director in extracted_preferences["directors"]["liked"]:
                            if isinstance(director, str) and director.lower() in item_directors:
                                item_score += 1.0

                        for director in extracted_preferences["directors"]["disliked"]:
                            if isinstance(director, str) and director.lower() in item_directors:
                                item_score -= 0.5

                        item_scores.append((item, item_score))

                        position_weight = 1.0 / (top_items.index(item) + 1)
                        preference_score += item_score * position_weight

                    if len(top_items) > 0:
                        preference_score = max(0, min(1, preference_score / len(top_items)))
                    else:
                        preference_score = 0

                    sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)

                    potential_attitude_score = 0

                    if sorted_items:
                        best_items = sorted_items[:3]

                        for item, score in best_items:

                            if score > 2.0:
                                potential_attitude_score += 0.3
                            elif score > 1.0:
                                potential_attitude_score += 0.2
                            elif score > 0.0:
                                potential_attitude_score += 0.1
                            else:
                                potential_attitude_score -= 0.5

                    genres_represented = set()
                    for item, _ in sorted_items[:5]:
                        if 'genre' in item:
                            if isinstance(item['genre'], str):
                                genres = [g.strip().lower() for g in item['genre'].split(',')]
                                genres_represented.update(genres)
                            elif isinstance(item['genre'], list):
                                for g in item['genre']:
                                    if isinstance(g, str):
                                        genres_represented.add(g.strip().lower())

                    diversity_bonus = min(0.2, len(genres_represented) * 0.05)
                    potential_attitude_score += diversity_bonus

                    combined_score = (0.3 * preference_score) + (0.7 * potential_attitude_score)

                    return combined_score, top_items

                except Exception as e:
                    logger.error(f"Error evaluating query: {str(e)}")
                    return 0, []

            def paraphrase_mcts(preferences, num_simulations=None):
                root = ParaphraseMCTSNode()
                num_simulations = num_simulations or len(paraphrase_strategies)

                for _ in range(num_simulations):
                    # Selection
                    node = root
                    while node.children and not node.is_terminal():
                        if len(node.children) < len(paraphrase_strategies):
                            # Not fully expanded
                            unexplored = [s for s in paraphrase_strategies if s not in node.children]
                            node.expand([unexplored[0]])
                            node = node.children[unexplored[0]]
                            break
                        else:
                            node = node.select_child()

                    # Expansion
                    if not node.is_terminal() and node.strategy is None:
                        children = node.expand(paraphrase_strategies)
                        if children:
                            node = random.choice(children)

                    # Simulation
                    if node.strategy:
                        # Generate query using this strategy
                        query = generate_retrieval_query(node.strategy, preferences)
                        # Evaluate the query quality
                        reward, _ = evaluate_query(query)
                        if self.trace:
                            self.trace.record(
                                "retrieval_reward",
                                strategy=node.strategy,
                                reward=reward,
                                query=query,
                            )
                    else:
                        reward = 0

                    # Backpropagation
                    while node is not None:
                        node.update(reward)
                        node = node.parent

                if not root.children:
                    return "balanced"

                best_child = max(root.children.values(), key=lambda n: n.value / max(n.visits, 1))
                return best_child.strategy

            best_strategy = paraphrase_mcts(extracted_preferences)
            if self.trace:
                self.trace.record("retrieval_strategy_selected", strategy=best_strategy)

            refined_query = generate_retrieval_query(best_strategy, extracted_preferences)

            conv_embed = call_embedding(refined_query).data[0].embedding
            conv_embed = np.asarray(conv_embed).reshape(1, -1)

            if len(self.item_emb_arr) == 0:
                raise StateError("No item embeddings available")

            sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
            rank_arr = np.argsort(sim_mat, axis=-1).tolist()
            rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
            item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
            item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

            self.item_to_recommend = []
            for i in range(min(5, len(item_rank_arr[0]))):
                try:
                    self.item_to_recommend.append(self.id2info[self.entityid2id[item_rank_arr[0][i]]])
                except (KeyError, IndexError):
                    continue

            meta_info = {
                'refined_query': refined_query,
                'strategy_used': best_strategy,
                'extracted_preferences': extracted_preferences
            }

            logger.info(f"Paraphrase strategy selected: {best_strategy}")
            logger.info(f"Refined query: {refined_query}")

            return meta_info, item_rank_arr

        except Exception as e:
            logger.error(f"Error in conversation paraphrase: {str(e)}")
            raise ActionExecutionError(f"Failed to execute conversation paraphrase: {str(e)}")

    def item_recommendation(self, conv_dict):
        """Recommend an item from the retrieved candidates."""
        if not self.item_to_recommend:
            self.conv_paraphrase(conv_dict)

        recommend_prompt = self.construct_recommend_prompt(conv_dict['context'])

        meta_info = {}
        gen_str = call_chatgpt_1(recommend_prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

        rec_item = ''
        rec_info = {'name': ''}
        for item in self.item_to_recommend:
            if item['name'] in gen_str and len(item['name']) > len(rec_item):
                rec_item = item['name']
                rec_info = item

        meta_info['rec_item'] = rec_item
        meta_info['rec_info'] = rec_info
        meta_info['strategies'] = None

        new_item = False
        if self.movie_info['name'] != rec_item:
            self.movie_info = {"name": rec_item}
            new_item = True
        if rec_item == '':
            new_item = True

        if gen_str.find('[REC]') != -1:
            ans_str = gen_str.split('[REC]')[0]
            meta_info['ans_type'] = 'rec'

            # Add to recommended items in state
            if rec_item and rec_item not in self.conversation_state['recommended_items']:
                self.conversation_state['recommended_items'].append(rec_item)
        else:
            ans_str = gen_str
            meta_info['ans_type'] = 'none'

        return meta_info, ans_str

    def item_explanation(self, conv_dict, simulate=False):
        """Provide an explanation for the recommended movie."""
        try:
            if not simulate:
                if not self.movie_info["name"]:
                    return self.get_conv(conv_dict)
                explanation_prompt = [
                {"role": "system", "content": f"""You are a movie expert.
                Provide an engaging explanation about the movie "{self.movie_info['name']}" according to the dialogue history.
                Include relevant details about the plot, actors, director, genre, and why this movie matches the user's preference.
                The explanation text you output should not exceed 100 words.
                Be informative but concise. End your response with the token [EXP]."""}
            ]
            else:
                if not self.movie_info_sim["name"]:
                    return self.get_conv(conv_dict)
                explanation_prompt = [
                {"role": "system", "content": f"""You are a movie expert.
                Provide an engaging explanation about the movie "{self.movie_info_sim['name']}".
                Include relevant details about the plot, actors, director, genre, and what makes this movie special.
                If the user further discusses the plot of the movie with you, you can chat with them by combining your own knowledge. If you don't know how to answer the user's question, you can also say that you don't know, and reiterate that your recommendation truly aligns with their preferences.
                The explanation text you output should not exceed 100 words.
                Do not repeat what you have already said earlier.
                Be informative but concise. End your response with the token [EXP].
                Dialogue History:\n
                """}
            ]

            for i, text in enumerate(conv_dict['context']):
                if not isinstance(text, str):
                    text = str(text)
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                explanation_prompt.append({
                    'role': role_str,
                    'content': text
                })

            gen_str = call_chatgpt_3(explanation_prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

            if gen_str.find('[EXP]') != -1:
                ans_str = gen_str.split('[EXP]')[0]
            else:
                ans_str = gen_str

            meta_info = {
                'ans_type': 'exp',
                'movie_info': ''
            }

            return meta_info, ans_str
        except Exception as e:
            logger.error(f"Error in item explanation: {str(e)}")
            raise ActionExecutionError(f"Failed to execute item explanation: {str(explanation_prompt)}")

    def failure_reflection(self, conv_dict, simulate=False):
        """Analyze the failure and determine the next best action to take."""
        try:
            analysis_prompt = [
                {"role": "system", "content": """You are a movie recommender system analyzing why your recommendations failed.
                Based on the conversation history, identify:
                1. The root cause of the failure (e.g., insufficient preferences, mismatched preferences, etc.)
                2. The most appropriate next action to take from these options:
                   - GenreInquiry: If we need more genre preferences, or the user has not expressed clearly the type they like.(e.g. The user says he like "thrilling and adrenaline-pumping movie", and you are not sure of if he is describing genre "action" or genre "thrilling".)
                   - ActorInquiry: If we need more actor preferences, like if the user has any other preferred actors.
                   - DirectorInquiry: If we need more director preferences, like if the user has any other preferred directors.
                   - ItemRecommendation: If all the user preference is explicit and we need to get better retrieval results.
                   - ItemExplanation: If the user misunderstands our recommendation of the movie, we need to explain to him that this movie actually fits his preferences.

                Tips: Do not keep asking the same attribute in several rounds.

                Return a JSON object with:
                {
                    "root_cause": "brief description of the failure",
                    "next_action": "one of the action types listed above",
                    "confidence": 0.0-1.0
                }"""}
            ]

            for i, text in enumerate(conv_dict['context']):
                if not isinstance(text, str):
                    text = str(text)
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                analysis_prompt.append({
                    'role': role_str,
                    'content': text
                })

            # Get the analysis from LLM
            analysis_json = call_chatgpt_3(analysis_prompt, model_name=GPT4, temperature=0.2, seed=0, json_mode=True)
            analysis = json.loads(analysis_json)

            # Validate the analysis
            if not isinstance(analysis, dict) or 'next_action' not in analysis:
                raise LLMError("Invalid analysis format from LLM")

            allowed_actions = ['GenreInquiry', 'ActorInquiry', 'DirectorInquiry', 'ItemRecommendation', 'ItemExplanation']
            raw_action = analysis['next_action']

            matches = difflib.get_close_matches(raw_action, allowed_actions, n=1, cutoff=0.6)
            if not matches:
                raise LLMError(f"Unrecognized action: {raw_action}")
            next_action = matches[0]
            meta_info = {
                'ans_type': 'reflection',
                'root_cause': analysis.get('root_cause', ''),
                'next_action': next_action,
                'confidence': analysis.get('confidence', 0.0)
            }
            if not simulate:
                self.conversation_state['actions_taken'].append(next_action)

            if next_action == "GenreInquiry":
                return self.genre_inquiry(conv_dict)
            elif next_action == "ActorInquiry":
                return self.actor_inquiry(conv_dict)
            elif next_action == "DirectorInquiry":
                return self.director_inquiry(conv_dict)
            elif next_action == "ItemExplanation":
                return self.item_explanation(conv_dict)
            elif next_action == "ItemRecommendation":
                conv_paraphrase, item_rank_arr = self.conv_paraphrase(conv_dict)
                a = {}
                b = ""
                c = conv_paraphrase.get('refined_query')
                a, b = self.reranking(conv_paraphrase)
                if not simulate:
                    self.conversation_state['actions_taken'].append("Retrieval")
                    self.conversation_state['actions_taken'].append("Reranking")
                    return item_rank_arr, a, b, c
                else:
                    return None, a, b, None

        except Exception as e:
            logger.error(f"Error in failure reflection: {str(e)}")
            raise ActionExecutionError(f"Failed to execute failure reflection: {str(e)}")

    def get_rec(self, conv_dict):
        """Original get_rec method - now used internally by the MCTS framework."""
        try:
            rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]

            context = conv_dict['context']
            context_list = []  # for model

            for i, text in enumerate(context):
                if not isinstance(text, str):
                    text = str(text)
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                context_list.append({
                    'role': role_str,
                    'content': text
                })

            conv_str = ""
            # import pdb
            # pdb.set_trace()

            for context in context_list:
                conv_str += f"{context['role']}: {context['content']} "

            conv_embed = call_embedding(conv_str).data[0].embedding
            conv_embed = np.asarray(conv_embed).reshape(1, -1)

            if len(self.item_emb_arr) == 0:
                raise StateError("No item embeddings available")

            sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
            rank_arr = np.argsort(sim_mat, axis=-1).tolist()
            rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
            item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
            item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

            self.item_to_recommend = []
            for i in range(5):
                self.item_to_recommend.append(self.id2info[self.entityid2id[item_rank_arr[0][i]]])

            return item_rank_arr, conv_str
        except Exception as e:
            logger.error(f"Error in get_rec: {str(e)}")
            raise ActionExecutionError(f"Failed to execute get_rec: {str(e)}")

    def get_conv(self, conv_dict):
        """Original get_conv method - now used as a fallback."""
        try:
            context = conv_dict['context']

            recommend_prompt = self.construct_recommend_prompt(context)

            meta_info = {}
            gen_str = call_chatgpt_1(recommend_prompt, model_name=GPT4, temperature=0.7, seed=0, json_mode=False)

            rec_item = ''
            rec_info = {'name': ''}
            for item in self.item_to_recommend:
                if item['name'] in gen_str and len(item['name']) > len(rec_item):
                    rec_item = item['name']
                    rec_info = item

            meta_info['rec_item'] = rec_item
            meta_info['rec_info'] = rec_info
            meta_info['strategies'] = None

            new_item = False
            if self.movie_info['name'] != rec_item:
                self.movie_info = {"name": rec_item}
                new_item = True
            if rec_item == '':
                new_item = True

            if gen_str.find('[EXP]') != -1 or (not new_item and gen_str.find('[REC]') == -1):
                ans_str = gen_str.split('[EXP]')[0]
                meta_info['ans_type'] = 'exp'
                meta_info['movie_info'] = rec_info
            elif gen_str.find('[REC]') != -1:
                ans_str = gen_str.split('[REC]')[0]
                meta_info['ans_type'] = 'rec'
            else:
                ans_str = gen_str
                meta_info['ans_type'] = 'none'

            return meta_info, ans_str
        except Exception as e:
            logger.error(f"Error in get_conv: {str(e)}")
            raise ActionExecutionError(f"Failed to execute get_conv: {str(e)}")

    def construct_recommend_prompt(self, context):
        """Original method to construct a recommendation prompt."""
        try:
            recommend_info = []
            for item in self.item_to_recommend:
                recommend_info.append({'name': item['name'], 'genre': item['genre'], 'star': item['star'], 'director': item['director']})

            prompt = f"""You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
1. If you do not have enough information about user preference, you should ask the user for his preference.
2. If you have enough information about user preference, you can give recommendation. If you decide to give recommendation, you should choose 1 item to recommend from the candidate list.
3. If the user ask you questions or make requests, you should respond to the user's request.
4. If you recommending a new movie for the first time in the dialogue history, add a special token '[REC]' at the end of your response.
5. If you are answering user's questions or telling details about the recommendation movie, give engaging descriptions and add a special token '[EXP]' at the end of your response.
6. Make sure your response is consistent with the given information, your response should honestly reflecting the given information and do not contain any deception.
7. Be brief in your response!

Candidate List
#######
"""
            for r in recommend_info:
                prompt += str(r['name']) + '\n'
            prompt += f"#######"
            messages = [{"role": "system", "content": prompt}]

            for i, text in enumerate(context):
                if not isinstance(text, str):
                    text = str(text)
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                messages.append({
                    'role': role_str,
                    'content': text
                })

            return messages
        except Exception as e:
            logger.error(f"Error in construct_recommend_prompt: {str(e)}")
            raise ActionExecutionError(f"Failed to construct recommend prompt: {str(e)}")

    def construct_inquiry_prompt(self, conv_dict, inquiry_type):
        """Construct a prompt for inquiring about user preferences."""
        try:
            if inquiry_type == 'genre':
                system_content = f"""You are a movie recommender chatting with a user.
            Your current task is to ask the user about their {inquiry_type} preferences in movies.
            Frame your question in a conversational and engaging way.
            Be specific and ask only about {inquiry_type} preferences.

            If the user has not expressed clearly the type they like, you should confirm it with them.(e.g. The user says he like "thrilling and adrenaline-pumping action movie", and you are not sure of if he is describing genre "action" or genre "thrilling". You may ask "I want to confirm. According to your description, do you like action movies?")

            Keep your response brief and focused.
            Here are the dialogue history:
            """
            elif inquiry_type == 'actor':
                system_content = f"""You are a movie recommender chatting with a user.
            Your current task is to ask the user about their {inquiry_type} preferences in movies.
            Frame your question in a conversational and engaging way.
            Be specific and ask only about {inquiry_type} preferences.

            If the user only mentioned one preferred star, and the star and his preferred director have never collaborated(Never fabricate a claim that two people who have collaborated have not collaborated!) you can try asking him about more preferred stars.

            Keep your response brief and focused.
            Here are the dialogue history:
            """
            elif inquiry_type == 'director':
                system_content = f"""You are a movie recommender chatting with a user.
            Your current task is to ask the user about their {inquiry_type} preferences in movies.
            Frame your question in a conversational and engaging way.
            Be specific and ask only about {inquiry_type} preferences.

            If the user only mentioned one preferred director, and the director and his preferred actors have never collaborated(Never fabricate a claim that two people who have collaborated have not collaborated!) you can try asking him about more preferred directors.

            Keep your response brief and focused.
            Here are the dialogue history:
            """

            messages = [{"role": "system", "content": system_content}]
            context = conv_dict['context']
            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                else:
                    role_str = 'assistant'
                messages.append({
                    'role': role_str,
                    'content': text
                })

            return messages
        except Exception as e:
            logger.error(f"Error in construct_inquiry_prompt: {str(e)}")
            raise ActionExecutionError(f"Failed to construct inquiry prompt: {str(e)}")

    @staticmethod
    def shuffle_user_preference(pref_dict):
    # Shuffle both keys and values
        keys = list(pref_dict.keys())
        random.shuffle(keys)
        new_pref = {}
        for key in keys:
            values = pref_dict[key]
            if isinstance(values, list):
                random.shuffle(values)
            new_pref[key] = values
        return new_pref


    def reranking(self, conv_dict, simulate=False):
        """Rerank the candidate items using LLM to select the best match for user preferences."""
        try:
            user_preference = ""
            if not isinstance(conv_dict, dict) or not conv_dict.get('refined_query'):
                user_preference = self.conversation_state.get('user_preferences', '')
                if not user_preference:
                    user_preference = self.conversation_state.get('context', {})
            else:
                user_preference = conv_dict['refined_query']

            if not self.item_to_recommend:
                raise ActionExecutionError(f"Failed to execute reranking: none item_to_recommend")

            candidates_info = []
            for item in self.item_to_recommend:
                candidate = {
                    'name': item['name'],
                    'genre': item['genre'],
                    'star': item['star'],
                    'director': item['director']
                }
                candidates_info.append(candidate)

            vote_counter = Counter()
            explanations = {}
            confidences = {}

            for i in range(3):
                shuffled_candidates = candidates_info.copy()
                random.shuffle(shuffled_candidates)
                shuffled_candidates_with_id = [
                {"id": f"C{i+1}", **item} for i, item in enumerate(shuffled_candidates)
                ]
                if isinstance(user_preference, dict):
                    shuffled_pref = self.shuffle_user_preference(user_preference)
                else:
                    shuffled_pref = user_preference

                rerank_prompt = [
                {"role": "system", "content": """You are a movie recommendation expert. Your task is to analyze the candidate movies and select the one that best matches the user's preferences based on the extracted preference.

                For each candidate, consider:
                1. Genre match with user's expressed preferences
                2. Actor/director match with user's preferences
                3. Overall fit with the conversation context

                The target movie must cater to all genres of movies that the user likes, and at least one star that the user prefers must be involved, as well as at least one director that the user likes must direct.
                You should only select one movie from the candidate list for one time.
                Not! Do not repeat recommending movies that you have recommended but the user does not like.

                Return a JSON object with:
                1. The selected movie name
                2. A confidence score (0-1)
                3. A brief explanation of why this movie was selected

                Format:
                {
                    "selected_movie": "movie name",
                    "confidence": 0.0-1.0,
                    "explanation": "brief explanation"
                }
                Here is the dialogue history:
                """}
                ]

                context = conv_dict.get('context', [])
                for i, text in enumerate(context):
                    if len(text) == 0:
                        continue
                    if i % 2 == 0:
                        role_str = 'user'
                    else:
                        role_str = 'assistant'
                    rerank_prompt.append({
                        'role': role_str,
                        'content': text
                    })
                rerank_prompt.append({
                    "role": "user",
                    "content": f"""
                    Here is the extracted user preference:\n{json.dumps(shuffled_pref, indent=2)}
                    Here are the candidate movies to analyze:\n{json.dumps(shuffled_candidates_with_id, indent=2)}"""
                })

            # Get LLM's analysis
                for attempt in range(3):
                    analysis_json = call_chatgpt_1(rerank_prompt, model_name=GPT4, temperature=0.2, seed=0, json_mode=True)
                    analysis = json.loads(analysis_json)

                    if not isinstance(analysis, dict) or 'selected_movie' not in analysis or 'confidence' not in analysis:
                        raise LLMError("Invalid analysis format from LLM")

                    selected_movie = analysis.get('selected_movie')
                    confidence = analysis.get('confidence', 0.0)
                    explanation = analysis.get('explanation', '')

                    found = False
                    for item in self.item_to_recommend:
                        if item['name'] == selected_movie:
                            self.movie_info_sim = {"name": selected_movie}
                            found = True
                            break
                    if found:
                        vote_counter[selected_movie] += 1
                        explanations.setdefault(selected_movie, []).append(explanation)
                        confidences.setdefault(selected_movie, []).append(confidence)
                        break
                    else:
                        rerank_prompt.append({
                            "role": "system",
                            "content": "You can only select from the candidate movies."
                        })
                else:
                    raise LLMError("LLM did not select a valid candidate after 3 attempts")

            if not vote_counter:
                raise ActionExecutionError("LLM failed to return any valid recommendation.")

            final_movie, _ = vote_counter.most_common(1)[0]
            avg_conf = sum(confidences[final_movie]) / len(confidences[final_movie])
            example_expl = explanations[final_movie][0]

            self.movie_info_sim = {"name": final_movie}
            if not simulate:
                self.movie_info = self.movie_info_sim

            if final_movie not in self.conversation_state['recommended_items']:
                self.conversation_state['recommended_items'].append(final_movie)

            # Prepare response
            meta_info = {
                'ans_type': 'rec',
                'rec_item': final_movie,
                'rec_info': self.movie_info,
                'confidence': avg_conf,
                'explanation': example_expl
            }

            response = f"I've carefully analyzed the options and I believe '{final_movie}' would be the best match for you. {example_expl}"

            return meta_info, response

        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            raise ActionExecutionError(f"Failed to execute reranking: {str(e)}")
