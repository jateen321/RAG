"""Unit tests for opt-in Gemini Google Search grounding."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rag_engine


class WebSearchTests(unittest.TestCase):
    def response(self):
        metadata = SimpleNamespace(
            grounding_chunks=[
                SimpleNamespace(web=SimpleNamespace(
                    uri="https://example.com/current-fact",
                    title="Example News",
                )),
            ],
            grounding_supports=[
                SimpleNamespace(
                    segment=SimpleNamespace(end_index=4),
                    grounding_chunk_indices=[0],
                ),
            ],
        )
        return SimpleNamespace(
            text="Fact.",
            candidates=[SimpleNamespace(grounding_metadata=metadata)],
        )

    def test_grounding_annotations_become_exact_web_citations(self):
        answer, sources = rag_engine._web_answer(self.response())

        self.assertEqual(answer, "Fact ⟦Web 1, Web⟧.")
        self.assertEqual(sources[0]["citation_label"], "Web 1")
        self.assertEqual(sources[0]["source_url"], "https://example.com/current-fact")
        self.assertEqual(sources[0]["source_type"], "web")

    def test_search_web_enables_google_search_only_on_the_explicit_call(self):
        model = Mock()
        model.models.generate_content.return_value = self.response()
        with patch.object(rag_engine, "_client", model):
            result = rag_engine.search_web("What happened today?")

        config = model.models.generate_content.call_args.kwargs["config"]
        self.assertTrue(config.tools)
        self.assertIsNotNone(config.tools[0].google_search)
        self.assertEqual(result["answer_basis"], "web")
        self.assertFalse(result["web_search_available"])


if __name__ == "__main__":
    unittest.main()
