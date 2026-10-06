import json
import numpy as np
import random
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import copy

from .action_mcts import MCTS
from .behavior import AgentBehaviorMixin
from .llm_client import CHAT_MODEL, call_chatgpt_2


GPT4 = CHAT_MODEL


class CHATGPT(AgentBehaviorMixin):

    def __init__(
        self,
        seed,
        debug,
        kg_dataset,
        mcts_iterations=3,
        simulation_workers=1,
        trace=None,
        repo_root=None,
    ) -> None:
        self.seed = seed
        self.simulation_workers = max(1, simulation_workers)
        self.debug = debug
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.kg_dataset = kg_dataset
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

        self.kg_dataset_path = self.repo_root / "data" / self.kg_dataset
        with open(self.kg_dataset_path / "entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(self.kg_dataset_path / "id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)

        # ReDial calls this field "star" while OpenDialKG calls it "actor".
        # Normalize once so the rest of the recommender uses one schema.
        for info in self.id2info.values():
            info.setdefault("star", info.get("actor", []))

        self.id2entity = {}
        for k, v in self.entity2id.items():
            self.id2entity[int(v)] = k

        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]

        self.entityid2id = {}
        for id, entityid in self.id2entityid.items():
            self.entityid2id[entityid] = id

        self.item_embedding_path = self.repo_root / "save" / "embed" / "item" / self.kg_dataset
        if not self.item_embedding_path.is_dir():
            raise FileNotFoundError(
                f"Missing item embeddings at {self.item_embedding_path}. "
                f"Run: python script/cache_item.py --dataset {self.kg_dataset}"
            )

        item_emb_list = []
        id2item_id = []
        for file in tqdm(sorted(self.item_embedding_path.iterdir()), desc="Loading item embeddings"):
            item_id = file.stem
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(file, encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
        self.item_to_recommend = []

        self.movie_info = {"name": ""}
        self.movie_info_sim = {"name": ""}

        self.mcts = MCTS(
            self,
            simulation_count=mcts_iterations,
            simulation_workers=simulation_workers,
            early_inquiry_rounds=2,
            trace=trace,
        )
        self.trace = trace
        if trace:
            trace.reset(dataset=kg_dataset, simulations=mcts_iterations, workers=simulation_workers)

        self.conversation_state = {
            'user_preferences': {
                'genres': [],
                'actors': [],
                'directors': []
            },
            'user_attitude': 'undecided',
            'recommended_items': [],
            'excluded_items': [],
            'actions_taken': [],
            'turn_count': 0,
            'context': []
        }

    def clone_for_simulation(self):
        """Copy mutable dialogue state while sharing read-only dataset arrays."""
        clone = copy.copy(self)
        clone.conversation_state = copy.deepcopy(self.conversation_state)
        clone.item_to_recommend = copy.deepcopy(self.item_to_recommend)
        clone.movie_info = copy.deepcopy(self.movie_info)
        clone.movie_info_sim = copy.deepcopy(self.movie_info_sim)
        clone.mcts = None
        return clone

    def select_action(self, conv_dict):
        """Use MCTS to select the next action."""

        self.update_conversation_state(conv_dict)
        best_action = self.mcts.search(self.conversation_state)

        return best_action


    def confidence_check(self, conv_dict):
        self.conversation_state['context'] = conv_dict['context']

        self.conversation_state['turn_count'] = sum(1 for i, text in enumerate(conv_dict['context'])
                                                  if i % 2 == 0 and len(text) > 0)

        context_list = []
        for i, text in enumerate(conv_dict['context']):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append(f"{role_str}: {text}")

        context = "\n".join(context_list)

        genre_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                      'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
                      'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                      'war', 'western']

        extraction_prompt = [
            {"role": "system", "content":
            """You are a movie preference analyzer. Extract the user's preference of movie genres according to the dialogue history.

            Genres should be from the following list: {'action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                    'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                    'war', 'western'}

            If you are unsure about the genres the user is describing as his preference, it means that clarification is needed.
            e.g. The user says he likes "kind that mixes thrill and adventure with some deep emotions", and you are not sure if he described an action movie or a thriller movie. There should be a clarification.

            Format your response as a JSON object with the following structure:
            {
                "need_clarification": Bool,
                "reason": ""
            }
            """
            },
            {"role": "user", "content": f"Analyze the following conversation:\n\n{context}"}
        ]

        try:
            # Call the LLM to extract preferences and attitude
            analysis_json = call_chatgpt_2(extraction_prompt, model_name=GPT4, temperature=0.1, seed=0, json_mode=True)
            analysis = json.loads(analysis_json)
            need_cqa = analysis.get('need_clarification', False)
            if need_cqa:
                reason = analysis.get('reason', "")
                cqa_prompt = [
                    {
                        "role": "system",
                        "content": f"Ask the user to clarify their preferred movie genre. Reason: {reason}",
                    },
                    {"role": "user", "content": "Ask one brief clarification question."},
                ]
                cqa = call_chatgpt_2(cqa_prompt, model_name=GPT4, temperature=0.1, seed=0, json_mode=False)
                return cqa
            else:
                return


        except Exception as e:
            logger.error(f"LLM analysis extraction failed: {e}")

            self._fallback_state_update(context)


    def update_conversation_state(self, conv_dict):
        """Update the conversation state with the latest context using LLM for preference extraction."""
        self.conversation_state['context'] = conv_dict['context']

        self.conversation_state['turn_count'] = sum(1 for i, text in enumerate(conv_dict['context'])
                                                  if i % 2 == 0 and len(text) > 0)
        context_list = []
        for i, text in enumerate(conv_dict['context']):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append(f"{role_str}: {text}")

        context = "\n".join(context_list)

        # Extract user preferences and attitude
        genre_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                      'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
                      'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                      'war', 'western']

        extraction_prompt = [
            {"role": "system", "content":
            """You are a movie preference and attitude analyzer. Extract information from the conversation.

            Identify:
            1. Movie preferences (genres, actors, directors)
            2. The latest USER message's attitude toward the current recommendation
            3. Movie titles the user has already seen or rejected (excluded_items)

            Preserve preferences stated earlier unless the user changes them. Only extract
            preferences expressed by the user; movie mentions or assistant suggestions
            alone are not positive preferences. Do not infer every genre of a mentioned film.
            Resolve actor/director clues to their names. The latest clarification overrides
            earlier assumptions. Judge attitude from the latest user message, not an older
            reaction or the assistant's enthusiasm. Never infer acceptance before a movie
            was recommended. Exclude watched/rejected films unless the user reconsiders.
            List changed_preferences only for categories the latest user explicitly
            corrects or replaces. Do not mark an unchanged category just because its
            values were not mentioned again. Adding another preference is not replacement.

            Genres should be from the following list: {'action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                    'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                    'war', 'western'}

            Here are some descriptions of movie genres that are easy to confuse:
            'action': 'thrilling and adrenaline-pumping action movie'
            'crime': 'suspenseful and intense criminal film'
            'mystery': 'intriguing and suspenseful mystery',
            'thriller': 'suspenseful and thrilling thriller with gripping suspense'
            The descriptions of these genres are very similar; you need to carefully distinguish which genre the user's description actually belongs to.

            Format your response as a JSON object with the following structure:
            {
                "preferences": {
                    "genres": ["genre1", "genre2", ...],
                    "actors": ["actor1", "actor2", ...],
                    "directors": ["director1", "director2", ...]
                },
                "attitude": "undecided | accepted | rejected",
                "attitude_confidence": 0.0-1.0,
                "excluded_items": ["catalogue movie titles, with year when known"],
                "changed_preferences": ["genres | actors | directors, only when replaced"]
            }

            For attitude classification:
            - "accepted" means the user has explicitly agreed to watch or shown strong positive interest in a specific recommended movie. Examples:
              * "Yes, I'll watch that movie!"
              * "That sounds perfect, I'll check it out"
              * "I love that movie, great choice!"
              * "I've been wanting to watch that one"

            - "rejected" means the user has explicitly declined or shown strong negative reaction to a specific recommended movie. Examples:
              * "No, I don't like that one"
              * "I've already seen that and didn't enjoy it"
              * "That's not really my style"
              * "I'd prefer something else"

            - "undecided" means:
              * The user is just expressing general preferences without specific movie reactions
              * The user is asking for recommendations without commenting on specific movies
              * The user's response is neutral or ambiguous
              * The user is still in the preference gathering phase

            Confidence should reflect how certain you are about the attitude assessment:
            - High confidence (>0.8): Clear explicit statements of acceptance/rejection
            - Medium confidence (0.5-0.8): Strong implicit signals
            - Low confidence (<0.5): Ambiguous or general statements

            Be conservative in attitude classification. When in doubt, default to "undecided".
            """
            },
            {"role": "user", "content": f"Analyze the following conversation:\n\n{context}"}
        ]

        try:
            analysis_json = call_chatgpt_2(extraction_prompt, model_name=GPT4, temperature=0.1, seed=0, json_mode=True)
            analysis = json.loads(analysis_json)

            preferences = analysis.get('preferences', {})
            for category in ('genres', 'actors', 'directors'):
                values = preferences.get(category)
                if isinstance(values, list):
                    previous = self.conversation_state['user_preferences'][category]
                    if category in analysis.get('changed_preferences', []):
                        previous = []
                    self.conversation_state['user_preferences'][category] = list(dict.fromkeys(
                        previous + [value.strip().lower() for value in values
                                    if isinstance(value, str) and value.strip()]))
            excluded = analysis.get('excluded_items')
            if isinstance(excluded, list):
                self.conversation_state['excluded_items'] = list(dict.fromkeys(
                    value.strip() for value in excluded if isinstance(value, str) and value.strip()))

            attitude = analysis.get('attitude', 'undecided')
            confidence = analysis.get('attitude_confidence', 0.0)

            if confidence > 0.4:  # Only update if confidence is reasonable
                self.conversation_state['user_attitude'] = attitude
            if not self.conversation_state['recommended_items']:
                self.conversation_state['user_attitude'] = 'undecided'
            elif self.conversation_state['user_attitude'] == 'rejected':
                name = self.movie_info.get('name')
                excluded = self.conversation_state.setdefault('excluded_items', [])
                if name and name not in excluded:
                    excluded.append(name)

            if self.debug:
                logger.info(f"Extracted analysis: {analysis}")
                logger.info(f"Updated conversation state: {self.conversation_state}")

        except Exception as e:
            logger.error(f"LLM analysis extraction failed: {e}")
            self._fallback_state_update(context)

    def _fallback_state_update(self, context_text):
        """Simple fallback method for updating state when LLM fails."""
        context_lower = context_text.lower()

        genres = ['action', 'comedy', 'drama', 'horror', 'sci-fi', 'romance', 'thriller', 'adventure']
        for genre in genres:
            if genre in context_lower and genre not in self.conversation_state['user_preferences']['genres']:
                self.conversation_state['user_preferences']['genres'].append(genre)

        actors = ['tom hanks', 'scarlett johansson', 'robert downey jr', 'emma stone', 'denzel washington']
        for actor in actors:
            if actor in context_lower and actor not in self.conversation_state['user_preferences']['actors']:
                self.conversation_state['user_preferences']['actors'].append(actor)

        directors = ['steven spielberg', 'christopher nolan', 'martin scorsese', 'quentin tarantino']
        for director in directors:
            if director in context_lower and director not in self.conversation_state['user_preferences']['directors']:
                self.conversation_state['user_preferences']['directors'].append(director)
