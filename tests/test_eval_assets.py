import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval" / "eval_set.json"
RESULTS = ROOT / "eval" / "results.json"


class EvaluationAssetTests(unittest.TestCase):
    def test_eval_set_has_expected_schema(self):
        with EVAL_SET.open(encoding="utf-8") as handle:
            items = json.load(handle)

        self.assertEqual(len(items), 15)

        for item in items:
            self.assertIn("question", item)
            self.assertIn("expected_answer_contains", item)
            self.assertTrue(item["question"].strip())
            self.assertTrue(item["expected_answer_contains"].strip())

    def test_saved_results_match_eval_count(self):
        with RESULTS.open(encoding="utf-8") as handle:
            results = json.load(handle)

        self.assertEqual(results["total_questions"], 15)
        self.assertEqual(len(results["results"]), 15)
        self.assertEqual(
            results["correct"] + sum(not item["correct"] for item in results["results"]),
            15,
        )


if __name__ == "__main__":
    unittest.main()
