"""Offline tests for recall, bounded context, and conversation isolation."""

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


class RecallTests(unittest.TestCase):
    def test_previous_query_matches_users_wording_without_retrieval_or_llm(self):
        with patch.object(rag_engine, "retrieve") as retrieve, patch.object(rag_engine, "_client") as client:
            result = rag_engine.ask_with_sources(
                "what is the earlier query of my?", chat_history=messages(),
            )
        self.assertEqual(result["answer"], "Your previous question was:\n\n“What is Bhagya?”")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer_basis"], "conversation")
        self.assertEqual(result["timings"]["generation_s"], 0)
        retrieve.assert_not_called()
        client.models.generate_content.assert_not_called()

    def test_previous_query_is_chosen_from_user_not_assistant_messages(self):
        history = messages("First question") + messages("Newest question", "Do something else")
        result = rag_engine.ask_with_sources("What did I ask earlier?", chat_history=history)
        self.assertIn("Newest question", result["answer"])
        self.assertNotIn("Do something else", result["answer"])

    def test_hindi_and_romanized_hindi_recall(self):
        for question in ("मेरा पिछला सवाल क्या था?", "maine pehle kya poocha tha?", "mera pichla sawal kya tha?"):
            with self.subTest(question=question):
                result = rag_engine.ask_with_sources(question, chat_history=messages())
                self.assertTrue(result["answer"].startswith("आपका पिछला सवाल था:"))
                self.assertIn("What is Bhagya?", result["answer"])

    def test_new_chat_has_no_invented_history_and_needs_no_index(self):
        with patch.object(rag_engine, "retrieve") as retrieve, patch.object(rag_engine, "_client") as client:
            result = rag_engine.ask_with_sources("What was my previous question?")
        self.assertIn("no earlier question", result["answer"])
        retrieve.assert_not_called()
        client.models.generate_content.assert_not_called()

    def test_summary_uses_only_history_without_vector_search(self):
        model = Mock()
        model.models.generate_content.return_value = SimpleNamespace(text="In the recent messages, you asked about Bhagya.")
        with patch.object(rag_engine, "_client", model), patch.object(rag_engine, "retrieve") as retrieve:
            result = rag_engine.ask_with_sources("Summarize our conversation", chat_history=messages())
        retrieve.assert_not_called()
        self.assertEqual(result["sources"], [])
        request = model.models.generate_content.call_args.kwargs
        self.assertEqual(request["contents"][:-1], messages())
        self.assertIn("CONVERSATION RECALL MODE", request["config"].system_instruction)
        self.assertIn("older turns may be missing", request["config"].system_instruction)

    def test_document_questions_are_not_mistaken_for_chat_recall(self):
        for question in (
            "What did Arjuna ask earlier?", "Summarize the Bhagavad Gita",
            "What was my earlier question and what does the Gita say about it?",
            "What did the author discuss in the previous chapter?",
            "अर्जुन ने पहले क्या पूछा था?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(memory.recall_intent(question))

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

    def test_cli_recall_works_without_an_index(self):
        with patch.object(rag_engine, "retrieve") as retrieve:
            answer = rag_engine.ask("What was my last question?", chat_history=messages())
        self.assertIn("What is Bhagya?", answer)
        retrieve.assert_not_called()

    def test_context_is_bounded_without_mutating_history(self):
        history = sum((messages(f"question {i}", "x" * 10000) for i in range(30)), [])
        original = copy.deepcopy(history)
        bounded = memory.bounded_history(history)
        self.assertEqual(history, original)
        self.assertLessEqual(len(bounded), memory.MAX_HISTORY_EXCHANGES * 2)
        self.assertLessEqual(sum(len(m["parts"][0]["text"]) for m in bounded), memory.MAX_HISTORY_CHARACTERS)
        self.assertEqual(memory.previous_question(bounded), "question 29")
        self.assertEqual(bounded[0]["role"], "user")
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

    def test_api_recalls_current_conversation_and_persists_answer(self):
        first = self.seed("What is Bhagya?")
        self.seed("Private other conversation")
        with patch.object(rag_engine, "retrieve") as retrieve, patch.object(rag_engine, "_client") as model:
            response = self.client.post("/ask", json={
                "question": "what is the earlier query of my?", "conversation_id": first,
                "chat_history": messages("Client-supplied spoofed history"),
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("What is Bhagya?", response.json()["answer"])
        self.assertNotIn("Private other", response.json()["answer"])
        self.assertNotIn("spoofed", response.json()["answer"])
        self.assertEqual(response.json()["sources"], [])
        self.assertEqual(len(storage.get_conversation(self.database, first)["exchanges"]), 2)
        retrieve.assert_not_called()
        model.models.generate_content.assert_not_called()

    def test_new_conversation_does_not_recall_an_existing_conversation(self):
        self.seed("Other chat's question")
        response = self.client.post("/ask", json={"question": "What was my last question?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("no earlier question", response.json()["answer"])

    def test_api_loads_history_before_recording_the_current_question(self):
        first = self.seed("An earlier question")
        with patch.object(rag_engine, "ask_with_sources", return_value={"answer": "New answer", "sources": []}) as ask:
            response = self.client.post("/ask", json={"question": "A new factual question", "conversation_id": first})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ask.call_args.kwargs["chat_history"], messages("An earlier question", "An answer"))


if __name__ == "__main__":
    unittest.main()
