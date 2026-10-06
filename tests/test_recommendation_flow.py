"""Regression checks for dialogue roles, constraints, and simulated state; no API calls."""
import copy
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
from src.model import action_mcts, agent, behavior
from src.model.errors import ActionExecutionError


def make_agent():
    a = agent.CHATGPT.__new__(agent.CHATGPT)
    a.conversation_state = {
        'user_preferences': {'genres': ['action', 'thriller'],
                             'actors': ['Denzel Washington'], 'directors': ['Tony Scott']},
        'user_attitude': 'undecided', 'recommended_items': [], 'excluded_items': [],
        'actions_taken': ['ActorInquiry', 'DirectorInquiry'], 'turn_count': 2,
        'context': ['u1', 'a1', 'u2', 'a2', 'latest user clarification'],
    }
    a.movie_info = {'name': ''}
    a.movie_info_sim = {'name': ''}
    a.trace = None
    a.debug = False
    a.seed = 100
    good = {'name': 'Suitable (2009)', 'genre': ['action', 'crime', 'thriller'],
            'star': ['Denzel Washington'], 'director': ['Tony Scott'],
            'plot': 'An ordinary dispatcher negotiates to protect hostages.'}
    wrong = {**good, 'name': 'Wrong Director (2012)', 'director': ['Someone Else']}
    a.item_to_recommend = [wrong, good]
    a.id2info = {str(i): {**wrong, 'name': f'Wrong {i}'} for i in range(6)}
    a.id2info['6'] = good
    a.id2item_id_arr = np.array(list(a.id2info))
    a.id2entityid = {str(i): i for i in range(7)}
    a.item_emb_arr = np.array([[1., 0.]]*6 + [[.1, 1.]])
    return a


class RecommendationFlowTest(unittest.TestCase):
    def test_latest_user_keeps_its_role_after_history_slice(self):
        a = make_agent()
        text = action_mcts.MCTS(a).format_state_for_llm(a.conversation_state)
        self.assertIn('Assistant: a2\nUser: latest user clarification', text)
        self.assertNotIn('Assistant: latest user clarification', text)

    def test_policy_cannot_explain_before_recommendation_or_skip_rejection(self):
        a = make_agent()
        bad_policy = json.dumps({'ItemExplanation': 10, 'ItemRecommendation': 9,
                                 'FailureReflection': 0, 'reason': 'bad prior'})
        with patch.object(action_mcts, 'call_chatgpt_1', return_value=bad_policy):
            scores, _ = action_mcts.MCTS(a).get_llm_action_policy(a.conversation_state)
            self.assertEqual(scores['ItemExplanation'], 0)
            a.conversation_state.update(recommended_items=['Suitable (2009)'], user_attitude='rejected')
            scores, _ = action_mcts.MCTS(a).get_llm_action_policy(a.conversation_state)
            self.assertEqual(scores['FailureReflection'], 1)
            self.assertEqual(scores['ItemRecommendation'], 0)

    def test_filter_enforces_known_people_and_all_genres_even_if_query_weakens_them(self):
        a = make_agent()
        wrong, good = a.item_to_recommend
        self.assertFalse(a._matches_preferences(wrong, {'directors': {'liked': []}}))
        self.assertFalse(a._matches_preferences({**good, 'genre': ['action']}, {}))
        self.assertFalse(a._matches_preferences(good, {'directors': {'disliked': ['Tony Scott']}}))
        a.conversation_state['excluded_items'] = ['Suitable']
        self.assertFalse(a._matches_preferences(good, {}))
        a.conversation_state['excluded_items'] = ['Suitable (1974)']
        self.assertTrue(a._matches_preferences(good, {}))

    def test_retrieval_does_not_lose_valid_item_below_unfiltered_top_five(self):
        a = make_agent()
        embedding = types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1., 0.])])
        with patch.object(behavior, 'call_embedding', return_value=embedding):
            ranks, items = a._retrieve_candidates('query', {})
        self.assertEqual(ranks, [[6]])
        self.assertEqual([i['name'] for i in items], ['Suitable (2009)'])

    def test_paraphrase_preserves_full_context(self):
        a = make_agent()
        preferences = {'genres': {'liked': ['action','thriller'], 'disliked': []},
                       'stars': {'liked': ['Denzel Washington'], 'disliked': []},
                       'directors': {'liked': ['Tony Scott'], 'disliked': []},
                       'themes': {'liked': ['hostages'], 'disliked': []}}
        context = list(a.conversation_state['context'])
        embedding = types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1., 0.])])
        with patch.object(behavior, 'call_chatgpt_1', side_effect=[json.dumps(preferences)]+['query']*5), \
             patch.object(behavior, 'call_embedding', return_value=embedding):
            meta, ranks = a.conv_paraphrase({'context': context})
        self.assertEqual(meta['context'], context)
        self.assertIsNot(meta['context'], context)
        self.assertEqual(ranks, [[6]])

    def test_reranking_rejects_invalid_choice_and_has_plot_and_history(self):
        a = make_agent()
        wrong, good = a.item_to_recommend
        bad = json.dumps({'selected_movie': wrong['name'], 'confidence': .9})
        valid = json.dumps({'selected_movie': good['name'], 'confidence': .9, 'explanation': 'plot fits'})
        with patch.object(behavior, 'call_chatgpt_1', side_effect=[bad,valid,valid,valid]) as call:
            meta, _ = a.reranking({'refined_query': 'query', 'context': a.conversation_state['context']})
        self.assertEqual(meta['rec_item'], good['name'])
        self.assertEqual(call.call_count, 4)
        prompt = call.call_args.args[0]
        self.assertIn({'role':'user','content':'latest user clarification'}, prompt)
        self.assertIn(good['plot'], prompt[-2]['content'] if prompt[-1]['role']=='system' else prompt[-1]['content'])
        self.assertNotIn(wrong['name'], prompt[-1]['content'])

    def test_no_eligible_candidate_never_asks_model_to_relax_constraints(self):
        a = make_agent()
        a.item_to_recommend = a.item_to_recommend[:1]
        with patch.object(behavior, 'call_chatgpt_1') as call:
            with self.assertRaisesRegex(ActionExecutionError, 'No catalogue candidates'):
                a.reranking({'refined_query':'query'})
        call.assert_not_called()

    def test_preference_correction_replaces_stale_values_and_tracks_rejection(self):
        a = make_agent()
        a.conversation_state['recommended_items'] = ['Suitable (2009)']
        a.movie_info = {'name':'Suitable (2009)'}
        analysis = {'preferences': {'genres':['comedy'], 'actors':['Denzel Washington'],
                                   'directors':['A New Director']},
                    'attitude':'rejected', 'attitude_confidence':.95, 'excluded_items':[],
                    'changed_preferences':['genres','directors']}
        with patch.object(agent, 'call_chatgpt_2', return_value=json.dumps(analysis)):
            a.update_conversation_state({'context':a.conversation_state['context']})
        self.assertEqual(a.conversation_state['user_preferences']['genres'], ['comedy'])
        self.assertEqual(a.conversation_state['user_preferences']['directors'], ['a new director'])
        self.assertIn('Suitable (2009)', a.conversation_state['excluded_items'])

    def test_unchanged_preferences_survive_partial_extraction(self):
        a = make_agent()
        analysis = {'preferences':{'genres':['action'],'actors':[],'directors':[]},
                    'attitude':'accepted','attitude_confidence':.95}
        with patch.object(agent, 'call_chatgpt_2', return_value=json.dumps(analysis)):
            a.update_conversation_state({'context':a.conversation_state['context']})
        self.assertEqual(a.conversation_state['user_preferences']['genres'], ['action','thriller'])
        self.assertEqual(a.conversation_state['user_preferences']['actors'], ['Denzel Washington'])
        self.assertEqual(a.conversation_state['user_attitude'], 'undecided')

    def test_simulation_recommendation_updates_only_cloned_state(self):
        a = make_agent()
        root_state = copy.deepcopy(a.conversation_state)
        a.movie_info = {'name':'Original'}
        a.movie_info_sim = {'name':'Original'}
        clone = a.clone_for_simulation()
        mcts = action_mcts.MCTS(a)
        good = a.item_to_recommend[1]
        valid = json.dumps({'selected_movie':good['name'], 'confidence':.9, 'explanation':'plot fits'})
        with patch.object(behavior, 'call_chatgpt_1', return_value=valid), \
             patch.object(agent.CHATGPT, 'conv_paraphrase', return_value=({'refined_query':'query'}, [[6]])), \
             patch.object(mcts, '_simulate_user_feedback', return_value='No thanks'), \
             patch.object(mcts, '_update_state_from_feedback', side_effect=lambda s,f:s.update(user_attitude='rejected')):
            simulated = mcts.apply_action(root_state, 'ItemRecommendation', clone)
        self.assertIn(good['name'], simulated['recommended_items'])
        self.assertIn(good['name'], simulated['excluded_items'])
        self.assertEqual(a.conversation_state, root_state)
        self.assertEqual(a.movie_info, {'name':'Original'})
        self.assertEqual(clone.movie_info, {'name':'Original'})
        self.assertEqual(clone.movie_info_sim['name'], good['name'])


if __name__ == '__main__':
    unittest.main()
