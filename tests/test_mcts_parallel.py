import copy
import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    sys.modules["loguru"] = types.SimpleNamespace(logger=logger)

from src.model.action_mcts import MCTS, MCTSNode


class FakeAgent:
    seed = 7

    def clone_for_simulation(self):
        return self


class DeterministicMCTS(MCTS):
    def simulate(self, node, simulation_id=None):
        return simulation_id + 1


class ParallelMCTSTest(unittest.TestCase):
    def test_parallel_batch_removes_virtual_loss_before_backpropagation(self):
        state = {
            "actions_taken": ["GenreInquiry", "ActorInquiry"],
            "turn_count": 2,
            "user_attitude": "undecided",
        }

        def transition(current_state, action):
            next_state = copy.deepcopy(current_state)
            next_state["actions_taken"].append(action)
            return next_state

        mcts = DeterministicMCTS(FakeAgent(), simulation_count=3, simulation_workers=3)
        root = MCTSNode(mcts.crs_agent, state, transition=transition)
        scores = {
            "GenreInquiry": 1,
            "ActorInquiry": 1,
            "DirectorInquiry": 1,
            "ItemRecommendation": 1,
            "ItemExplanation": 1,
            "FailureReflection": 1,
        }
        mcts._run_parallel(root, scores, state)

        self.assertEqual(root.visits, 3)
        self.assertEqual(root.value, 6)
        self.assertEqual(sum(child.visits for child in root.children.values()), 3)


if __name__ == "__main__":
    unittest.main()
