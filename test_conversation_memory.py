"""Offline tests for bounded context and conversation isolation."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import api
import conversation_memory as memory
import conversation_store as storage
import rag_engine


def messages(question="What is Bhagya?", answer="Earlier answer"):
    return [
        {"role": "user", "parts": [{"text": question}]},
        {"role": "model", "parts": [{"text": answer}]},
    ]


class ConversationMemoryTests(unittest.TestCase):
    def test_context_dependent_question_is_rewritten_before_retrieval(self):
        chunk = {"text": "A verified passage", "page": 1, "source": "book.pdf", "distance": 0.1}
        model = Mock()
        model.models.generate_content.side_effect = [
            SimpleNamespace(text='{"standalone_query":"What does the Bhagya book say?"}'),
            SimpleNamespace(text="Grounded answer"),
        ]
        with patch.object(rag_engine, "retrieve", return_value=[chunk]) as retrieve, patch.object(rag_engine, "_client", model):
            result = rag_engine.ask_with_sources(
                "What does this book say?", top_k=3, chat_history=messages(),
            )
        retrieve.assert_called_once_with("What does the Bhagya book say?", top_k=3)
        generation_request = model.models.generate_content.call_args_list[1].kwargs
        self.assertEqual(generation_request["contents"][:-1], messages())
        self.assertEqual(result["sources"][0]["source"], "book.pdf")
        self.assertTrue(result["retrieval"]["contextualized"])
        self.assertEqual(result["retrieval"]["query"], "What does the Bhagya book say?")
        self.assertIn("previous assistant claims are not verified evidence", rag_engine.SYSTEM_PROMPT)

    def test_self_contained_question_skips_contextualizer(self):
        chunk = {"text": "A verified passage", "page": 1, "source": "book.pdf", "distance": 0.1}
        model = Mock()
        model.models.generate_content.return_value = SimpleNamespace(text="Grounded answer")
        with patch.object(rag_engine, "retrieve", return_value=[chunk]) as retrieve, patch.object(rag_engine, "_client", model):
            result = rag_engine.ask_with_sources(
                "What is the meaning of Bhagya?", chat_history=messages(),
            )
        retrieve.assert_called_once_with("What is the meaning of Bhagya?")
        self.assertEqual(model.models.generate_content.call_count, 1)
        self.assertFalse(result["retrieval"]["contextualized"])

    def test_contextualizer_failure_falls_back_to_original_question(self):
        chunk = {"text": "A verified passage", "page": 1, "source": "book.pdf", "distance": 0.1}
        model = Mock()
        model.models.generate_content.side_effect = [
            SimpleNamespace(text="not-json"),
            SimpleNamespace(text="Grounded answer"),
        ]
        with patch.object(rag_engine, "retrieve", return_value=[chunk]) as retrieve, patch.object(rag_engine, "_client", model):
            result = rag_engine.ask_with_sources(
                "What does this book say?", chat_history=messages(),
            )
        retrieve.assert_called_once_with("What does this book say?")
        self.assertFalse(result["retrieval"]["contextualized"])

    def test_bilingual_follow_up_detection(self):
        history = messages()
        for question in (
            "What happened after that?", "इसके बाद क्या हुआ?", "phir kya hua?",
        ):
            with self.subTest(question=question):
                self.assertTrue(memory.needs_contextualization(question, history))
        self.assertFalse(memory.needs_contextualization("Who was Ashoka?", history))
        self.assertFalse(memory.needs_contextualization("What happened after that?", []))

    def test_context_is_bounded_without_mutating_history(self):
        history = sum((messages(f"question {i}", "x" * 10000) for i in range(30)), [])
        original = copy.deepcopy(history)
        bounded = memory.bounded_history(history)
        self.assertEqual(history, original)
        self.assertLessEqual(len(bounded), memory.MAX_HISTORY_EXCHANGES * 2)
        self.assertLessEqual(sum(len(m["parts"][0]["text"]) for m in bounded), memory.MAX_HISTORY_CHARACTERS)
        self.assertEqual(bounded[0]["role"], "user")
        self.assertEqual(bounded[-2]["parts"][0]["text"], "question 29")
        self.assertIn("[message shortened]", bounded[-1]["parts"][0]["text"])


class StoredHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "conversations.sqlite3"
        self.enterContext(patch.object(api, "CONVERSATION_DB_PATH", self.database))
        self.client = TestClient(api.app)

    def seed(self, question, conversation_id=None):
        return storage.record_exchange(self.database, conversation_id, question, "An answer", [], 0.1)

    def test_history_query_is_scoped_bounded_and_ordered_on_timestamp_ties(self):
        with patch.object(storage, "_now", return_value="2026-08-27T00:00:00+00:00"):
            first = self.seed("first")
            self.seed("second", first)
            self.seed("third", first)
            self.seed("unrelated conversation")
        history = storage.get_recent_history(self.database, first, limit=2)
        self.assertEqual([m["parts"][0]["text"] for m in history if m["role"] == "user"], ["second", "third"])
        self.assertEqual(storage.get_recent_history(self.database, "missing"), [])

    def test_api_loads_history_before_recording_the_current_question(self):
        first = self.seed("An earlier question")
        with patch.object(rag_engine, "ask_with_sources", return_value={"answer": "New answer", "sources": []}) as ask:
            response = self.client.post("/ask", json={"question": "A new factual question", "conversation_id": first})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ask.call_args.kwargs["chat_history"], messages("An earlier question", "An answer"))


if __name__ == "__main__":
    unittest.main()
