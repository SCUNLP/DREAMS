import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from src.model.parallel import iter_completed


class ParallelTest(unittest.TestCase):
    def test_parallel_jobs_keep_their_results_attached(self):
        results = dict(iter_completed(lambda value: value * value, [1, 2, 3, 4], 2))
        self.assertEqual(results, {1: 1, 2: 4, 3: 9, 4: 16})

    def test_single_worker_uses_the_same_contract(self):
        self.assertEqual(list(iter_completed(str, [1, 2], 1)), [(1, "1"), (2, "2")])


if __name__ == "__main__":
    unittest.main()
