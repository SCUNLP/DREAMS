"""Offline checks for independent calls, query reuse, and explanation-only decisions."""
import json
import copy
import threading
import unittest
from unittest.mock import patch

from test_recommendation_flow import make_agent
from src.model import behavior, action_mcts
from src.model.demo import DemoTrace


class DemoSpeedTest(unittest.TestCase):
    def test_strategy_evaluations_overlap_and_winner_is_not_regenerated(self):
        agent = make_agent()
        agent.simulation_workers = 3
        barrier = threading.Barrier(3)
        preferences = {k: {'liked':[], 'disliked':[]} for k in ('genres','stars','directors','themes')}
        good = agent.item_to_recommend[-1]

        def chat(prompt, **kwargs):
            system = prompt[0]['content']
            if 'Current strategy:' not in system:
                return json.dumps(preferences)
            strategy = system.split('Current strategy: ')[1].split()[0]
            if strategy != 'positive_only':
                barrier.wait(timeout=5)
            return strategy+' query'

        with patch.object(behavior, 'call_chatgpt_1', side_effect=chat) as calls, \
             patch.object(agent, '_retrieve_candidates', return_value=([[6]], [good])) as retrieval:
            meta, ranks = agent.conv_paraphrase({'context':['user preference']})
        self.assertEqual(calls.call_count, 5)  # one extraction, four queries
        self.assertEqual(retrieval.call_count, 4)  # no repeated winning embedding
        self.assertEqual(meta['strategy_used'], 'balanced')
        self.assertEqual(meta['refined_query'], 'balanced query')
        self.assertEqual(ranks, [[6]])

    def test_rerank_votes_overlap_and_only_mutate_state_after_all_votes(self):
        agent = make_agent()
        agent.simulation_workers = 3
        barrier = threading.Barrier(3)
        good = agent.item_to_recommend[-1]

        def chat(prompt, **kwargs):
            self.assertEqual(agent.conversation_state['recommended_items'], [])
            barrier.wait(timeout=5)
            return json.dumps({'selected_movie':good['name'], 'confidence':.9, 'explanation':'plot fits'})

        with patch.object(behavior, 'call_chatgpt_1', side_effect=chat) as calls:
            meta, _ = agent.reranking({'refined_query':'query'})
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(meta['rec_item'], good['name'])
        self.assertEqual(agent.conversation_state['recommended_items'], [good['name']])

    def test_rollout_starts_before_the_next_expansion_finishes(self):
        started = threading.Event()
        agent = make_agent()

        class PipelineMCTS(action_mcts.MCTS):
            selections = 0

            def select(self, root, scores):
                if self.selections == 1:
                    if not started.wait(timeout=5):
                        raise AssertionError('First rollout was delayed until all expansions finished')
                self.selections += 1
                return root.expand(scores)

            def simulate(self, node, simulation_id=None):
                started.set()
                return simulation_id+1

        mcts = PipelineMCTS(agent, simulation_count=3, simulation_workers=3)
        root = action_mcts.MCTSNode(agent, copy.deepcopy(agent.conversation_state),
                                   transition=lambda state,action: state)
        mcts._run_parallel(root, {'GenreInquiry':1,'ActorInquiry':1,'DirectorInquiry':1}, root.state)
        self.assertEqual(root.visits, 3)
        self.assertEqual(root.value, 6)

    def test_only_explanation_skips_rollouts_and_trace_reports_zero_visits(self):
        agent = make_agent()
        trace = DemoTrace(None, auto_open=False)
        mcts = action_mcts.MCTS(agent, trace=trace)
        with patch.object(mcts, 'get_llm_action_policy', return_value=({'ItemExplanation':1.0}, 'explain')), \
             patch.object(mcts, '_run_sequential') as search:
            self.assertEqual(mcts.search(agent.conversation_state), 'ItemExplanation')
        search.assert_not_called()
        self.assertEqual(trace.snapshot()['tree']['visits'], 0)
        self.assertIn('forced_action', [event['event'] for event in trace.events])
        for action in ('ItemRecommendation','FailureReflection'):
            with patch.object(mcts, 'get_llm_action_policy', return_value=({action:1.0}, 'search')), \
                 patch.object(mcts, '_run_sequential') as search:
                self.assertEqual(mcts.search(agent.conversation_state), action)
            search.assert_called_once()


if __name__ == '__main__':
    unittest.main()
