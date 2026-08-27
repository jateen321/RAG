"""Scoring-logic tests for evaluate.py.

The retriever is stubbed out, so these run offline and cost no embedding quota.
They exist because the v2 schema changed how a "hit" is defined in three ways
(source lists, match modes, inverted unanswerable scoring) and a silent
mis-scoring would look like a retrieval result rather than a harness bug.
"""

import sys
import types
import unittest

_fake = types.ModuleType("retriever")
_STORE: dict[str, list[dict]] = {}
_fake.retrieve = lambda q, top_k=None: _STORE.get(q, [])[:top_k]
_previous_retriever = sys.modules.get("retriever")
sys.modules["retriever"] = _fake
try:
    import evaluate as E  # noqa: E402
finally:
    if _previous_retriever is None:
        sys.modules.pop("retriever", None)
    else:
        sys.modules["retriever"] = _previous_retriever


def chunk(source, page, distance=0.2):
    return {"source": source, "page": page, "distance": distance, "text": "x"}


class ExpectedSourcesTests(unittest.TestCase):
    def test_list_matches_either_duplicate_spelling(self):
        # Three documents are indexed under both 'X.pdf' and 'data/X.pdf'.
        q = {"match": "page", "expected_sources": ["X.pdf", "data/X.pdf"],
             "expected_pages": [7]}
        self.assertEqual(E._rank_expected_source(q, [chunk("data/X.pdf", 7)]), 1)

    def test_v1_singular_key_still_supported(self):
        self.assertEqual(E._expected_sources({"expected_source": "Y.pdf"}), ["Y.pdf"])


class MatchModeTests(unittest.TestCase):
    def test_source_mode_ignores_meaningless_page_number(self):
        # Every .txt chunk carries page_number=1, so page equality proves nothing.
        q = {"match": "source", "expected_sources": ["M.txt"], "expected_pages": []}
        self.assertEqual(E._rank_expected_source(q, [chunk("M.txt", 1)]), 1)

    def test_page_mode_rejects_wrong_page(self):
        q = {"match": "page", "expected_sources": ["X.pdf"], "expected_pages": [7]}
        self.assertIsNone(E._rank_expected_source(q, [chunk("X.pdf", 99)]))

    def test_rank_is_one_based_position(self):
        q = {"match": "source", "expected_sources": ["M.txt"], "expected_pages": []}
        self.assertEqual(
            E._rank_expected_source(q, [chunk("Other.pdf", 3), chunk("M.txt", 1)]), 2
        )

    def test_none_mode_never_claims_a_hit(self):
        q = {"match": "none", "expected_sources": [], "expected_pages": []}
        self.assertIsNone(E._rank_expected_source(q, [chunk("Any.pdf", 1)]))


class RefusalDetectionTests(unittest.TestCase):
    def test_detects_english_and_hindi_refusals(self):
        for text in ("The provided context does not contain this information.",
                     "दिए गए संदर्भ में इसकी जानकारी नहीं है।",
                     "पर्याप्त जानकारी नहीं है।"):
            self.assertTrue(E._is_refusal(text), text)

    def test_does_not_flag_ordinary_answers(self):
        for text in ("Registration opens on 12-05-2026 (CIL.pdf, Page 1).",
                     "गीता का रचनाकाल पांच सहस्र वर्ष पूर्व माना जाता है।"):
            self.assertFalse(E._is_refusal(text), text)


class UnanswerableScoringTests(unittest.TestCase):
    def test_unanswerable_rows_leave_the_hit_rate_denominator(self):
        _STORE.clear()
        _STORE["good"] = [chunk("X.pdf", 7)]
        _STORE["bad"] = [chunk("Noise.pdf", 1)]
        dataset = [
            {"id": "a", "language": "en", "category": "factual", "match": "page",
             "question": "good", "expected_sources": ["X.pdf"],
             "expected_pages": [7], "answer_keywords": []},
            {"id": "u", "language": "en", "category": "unanswerable", "match": "none",
             "question": "bad", "expected_sources": [], "expected_pages": [],
             "answer_keywords": []},
        ]
        report = E.evaluate(dataset, top_k=5)
        summary = report["summary"]
        self.assertEqual(summary["scored_questions"], 1)
        self.assertEqual(summary["unanswerable_questions"], 1)
        # Would be 0.5 if the unanswerable row were counted as a miss.
        self.assertEqual(summary["retrieval_hit_rate"], 1.0)
        row = next(r for r in report["results"] if r["id"] == "u")
        self.assertIsNone(row["retrieval_hit"])


if __name__ == "__main__":
    unittest.main()
