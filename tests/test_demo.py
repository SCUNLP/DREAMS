import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from src.model.demo import DemoTrace


class DemoTraceTest(unittest.TestCase):
    def test_render_contains_tree_reward_and_feedback(self):
        child = SimpleNamespace(action="GenreInquiry", visits=1, value=0.5, children={})
        root = SimpleNamespace(action=None, visits=1, value=0.5, children={"GenreInquiry": child})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo.html"
            trace = DemoTrace(output, auto_open=False)
            trace.record("feedback", reward=0.5, feedback="I like comedy")
            trace.render(root, {"turn_count": 3})
            document = output.read_text(encoding="utf-8")
        self.assertIn("GenreInquiry", document)
        self.assertIn("I like comedy", document)
        self.assertIn("Reward and feedback timeline", document)

    def test_memory_trace_returns_detached_snapshot(self):
        trace = DemoTrace(None, auto_open=False)
        trace.record("feedback", reward=0.5)
        snapshot = trace.render(state={"turn_count": 2})
        snapshot["state"]["turn_count"] = 99
        snapshot["events"].append({"event": "external"})
        current = trace.snapshot()
        self.assertEqual(current["state"]["turn_count"], 2)
        self.assertEqual(len(current["events"]), 1)


if __name__ == "__main__":
    unittest.main()
