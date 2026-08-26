import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from src.model.early_policy import select_early_inquiry


class EarlyPolicyTest(unittest.TestCase):
    def test_only_inquiry_actions_are_considered(self):
        scores = {
            "GenreInquiry": 0.2,
            "ActorInquiry": 0.3,
            "DirectorInquiry": 0.1,
            "ItemRecommendation": 0.9,
        }
        self.assertEqual(select_early_inquiry(scores), "ActorInquiry")

    def test_ties_start_with_genre_inquiry(self):
        self.assertEqual(select_early_inquiry({}), "GenreInquiry")

    def test_second_turn_does_not_repeat_the_first_inquiry(self):
        scores = {"GenreInquiry": 0.9, "ActorInquiry": 0.5, "DirectorInquiry": 0.2}
        self.assertEqual(select_early_inquiry(scores, ["GenreInquiry"]), "ActorInquiry")


if __name__ == "__main__":
    unittest.main()
