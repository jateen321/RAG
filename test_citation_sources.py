"""Regression tests for citation/source alignment across follow-up turns."""

import tempfile
import unittest
from pathlib import Path

from citation_sources import align_answer_sources
from conversation_memory import needs_contextualization
from conversation_store import get_conversation, get_recent_history, record_exchange


class CitationSourceAlignmentTests(unittest.TestCase):
    def test_historical_sources_replace_unrelated_follow_up_retrieval(self):
        current = [{
            "chunk_id": "milk",
            "source": "YouTube: Milk lesson [abc]",
            "timestamp": "16:46",
            "source_type": "youtube",
        }]
        historical = [
            {
                "chunk_id": "health-page",
                "page": 228,
                "source": "सम्पूर्ण-चाणक्य-नीति.pdf",
                "source_type": "pdf",
            },
            {
                "chunk_id": "health-video",
                "source": "YouTube: Health vlog [xyz]",
                "video_title": "Health vlog",
                "timestamp": "8:30",
                "source_type": "youtube",
            },
        ]
        answer = (
            "स्वास्थ्य सलाह ⟦सम्पूर्ण-चाणक्य-नीति.pdf, पृष्ठ 228 / Page 228⟧ "
            "और अभ्यास ⟦Health vlog, Timestamp 8:30⟧।"
        )

        self.assertEqual(
            [source["chunk_id"] for source in align_answer_sources(answer, current, historical)],
            ["health-page", "health-video"],
        )

    def test_current_sources_remain_when_answer_has_no_matchable_citation(self):
        current = [{"chunk_id": "current", "source": "book.pdf", "page": 1}]

        self.assertEqual(align_answer_sources("Answer without a citation.", current), current)

    def test_language_only_instruction_is_a_context_dependent_follow_up(self):
        history = [{"role": "user", "parts": [{"text": "How can I improve my health?"}]}]

        self.assertTrue(needs_contextualization("explain in hindia", history))

    def test_stored_follow_up_recovers_cited_sources_from_prior_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "conversations.sqlite3"
            source = {
                "chunk_id": "health-page",
                "page": 228,
                "source": "health.pdf",
                "source_type": "pdf",
            }
            conversation_id = record_exchange(
                database,
                None,
                "How can I improve my health?",
                "Advice ⟦health.pdf, Page 228⟧.",
                [source],
                0.1,
            )
            record_exchange(
                database,
                conversation_id,
                "explain in hindia",
                "सलाह ⟦health.pdf, पृष्ठ 228 / Page 228⟧।",
                [{
                    "chunk_id": "unrelated",
                    "source": "milk-video",
                    "timestamp": "16:46",
                    "source_type": "youtube",
                }],
                0.1,
            )

            conversation = get_conversation(database, conversation_id)
            self.assertEqual(
                [item["chunk_id"] for item in conversation["exchanges"][1]["sources"]],
                ["health-page"],
            )
            history = get_recent_history(database, conversation_id)
            self.assertEqual(history[1]["sources"], [source])


if __name__ == "__main__":
    unittest.main()
