import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import (  # noqa: E402
    is_follow_up_query,
    reciprocal_rank_fusion,
    select_final_context,
    tokenize,
)


class RagUtilityTests(unittest.TestCase):
    def test_tokenize_removes_common_stop_words(self):
        tokens = tokenize("What is the difference between interval and ratio data?")

        self.assertIn("difference", tokens)
        self.assertIn("interval", tokens)
        self.assertIn("ratio", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("between", tokens)
        self.assertNotIn("and", tokens)

    def test_follow_up_requires_previous_question(self):
        self.assertFalse(is_follow_up_query("Explain it", None))
        self.assertTrue(
            is_follow_up_query(
                "Explain it",
                "What is an operating system?",
            )
        )
        self.assertFalse(
            is_follow_up_query(
                "What is BM25?",
                "What is an operating system?",
            )
        )

    def test_rrf_merges_rankings(self):
        dense = [
            {"source": "a.pdf", "chunk_index": 0, "text": "alpha", "dense_rank": 1},
            {"source": "b.pdf", "chunk_index": 0, "text": "beta", "dense_rank": 2},
        ]
        lexical = [
            {"source": "b.pdf", "chunk_index": 0, "text": "beta", "lexical_rank": 1},
            {"source": "c.pdf", "chunk_index": 0, "text": "gamma", "lexical_rank": 2},
        ]

        fused = reciprocal_rank_fusion(dense, lexical, k=3)

        self.assertEqual(fused[0]["text"], "beta")
        self.assertEqual(len(fused), 3)

    def test_context_selection_limits_duplicate_text_and_source(self):
        candidates = [
            {"source": "a.pdf", "text": "same", "reranker_score": 1.0},
            {"source": "a.pdf", "text": "same", "reranker_score": 0.9},
            {"source": "a.pdf", "text": "other", "reranker_score": 0.8},
            {"source": "a.pdf", "text": "third", "reranker_score": 0.7},
            {"source": "b.pdf", "text": "different", "reranker_score": 0.6},
        ]

        selected = select_final_context(candidates, k=6)

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            [item["text"] for item in selected],
            ["same", "other", "different"],
        )
        self.assertLessEqual(
            sum(item["source"] == "a.pdf" for item in selected),
            2,
        )


if __name__ == "__main__":
    unittest.main()
