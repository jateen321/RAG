"""Offline tests for bounded context and conversation isolation."""

import copy
import json
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
from auth import AuthenticatedUser, get_current_user


def messages(question="What is Bhagya?", answer="Earlier answer"):
    return [
        {"role": "user", "parts": [{"text": question}]},
        {"role": "model", "parts": [{"text": answer}]},
    ]


def document_response(answer="Grounded answer", insufficient=False):
    return SimpleNamespace(text=json.dumps({
        "answer": answer,
        "insufficient_information": insufficient,
    }))


def retrieval_result(chunks):
    return {
        "chunks": chunks,
        "queries": ["query"],
        "raw_candidate_count": len(chunks),
        "unique_candidate_count": len(chunks),
        "rerank_candidate_count": len(chunks),
        "timings": {},
    }


class ConversationMemoryTests(unittest.TestCase):
    def test_context_dependent_question_is_rewritten_before_retrieval(self):
        chunk = {"text": "A verified passage", "page": 1, "source": "book.pdf", "distance": 0.1}
        model = Mock()
        model.models.generate_content.side_effect = [
            SimpleNamespace(text='{"standalone_query":"What does the Bhagya book say?"}'),
            document_response(),
        ]
        with patch.object(rag_engine, "retrieve_context", return_value=retrieval_result([chunk])) as retrieve, patch.object(rag_engine, "_client", model):
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
        model.models.generate_content.return_value = document_response()
        with patch.object(rag_engine, "retrieve_context", return_value=retrieval_result([chunk])) as retrieve, patch.object(rag_engine, "_client", model):
            result = rag_engine.ask_with_sources(
                "What is the meaning of Bhagya?", chat_history=messages(),
            )
        retrieve.assert_called_once_with("What is the meaning of Bhagya?")
        self.assertEqual(model.models.generate_content.call_count, 1)
        self.assertFalse(result["retrieval"]["contextualized"])

    def test_image_prompt_uses_retrieved_evidence_instead_of_answer(self):
        chunk = {
            "text": "Chlorophyll absorbs light used in photosynthesis.",
            "page": 7,
            "source": "biology.pdf",
            "distance": 0.1,
        }
        model = Mock()
        model.models.generate_content.return_value = document_response(
            "A model-written summary that must not ground the image."
        )
        with (
            patch.object(
                rag_engine,
                "retrieve_context",
                return_value=retrieval_result([chunk]),
            ),
            patch.object(rag_engine, "_client", model),
        ):
            result = rag_engine.ask_with_sources(
                "Show how photosynthesis works.",
                prepare_image_prompt=True,
            )

        prompt = result["image_prompt"]
        self.assertIn("Student request: Show how photosynthesis works.", prompt)
        self.assertIn("biology.pdf · पृष्ठ 7 / Page 7", prompt)
        self.assertIn("Chlorophyll absorbs light used in photosynthesis.", prompt)
        self.assertNotIn("model-written summary", prompt)

    def test_insufficient_document_answer_does_not_enable_web_implicitly(self):
        chunk = {"text": "An unrelated passage", "page": 1, "source": "book.pdf", "distance": 0.9}
        model = Mock()
        model.models.generate_content.return_value = document_response(
            "The indexed sources do not contain enough information.", True,
        )
        with patch.object(rag_engine, "retrieve_context", return_value=retrieval_result([chunk])), patch.object(rag_engine, "_client", model):
            result = rag_engine.ask_with_sources("What happened today?")

        self.assertTrue(result["insufficient_information"])
        self.assertFalse(result["web_search_available"])
        self.assertEqual(result["answer_basis"], "documents")
        config = model.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_schema, rag_engine._ANSWER_SCHEMA)

    def test_contextualizer_failure_falls_back_to_original_question(self):
        chunk = {"text": "A verified passage", "page": 1, "source": "book.pdf", "distance": 0.1}
        model = Mock()
        model.models.generate_content.side_effect = [
            SimpleNamespace(text="not-json"),
            document_response(),
        ]
        with patch.object(rag_engine, "retrieve_context", return_value=retrieval_result([chunk])) as retrieve, patch.object(rag_engine, "_client", model):
            result = rag_engine.ask_with_sources(
                "What does this book say?", chat_history=messages(),
            )
        retrieve.assert_called_once_with("What does this book say?")
        self.assertFalse(result["retrieval"]["contextualized"])

    def test_bilingual_follow_up_detection(self):
        history = messages()
        for question in (
            "What happened after that?", "इसके बाद क्या हुआ?", "phir kya hua?",
            "explain in hindia", "translate this in Hindi",
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
        api.app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            uid=storage.LOCAL_OWNER_ID, email="local@example.com", is_admin=True
        )
        self.addCleanup(api.app.dependency_overrides.clear)
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

    def test_web_search_replaces_latest_eligible_exchange(self):
        exchange_id = "eligible-exchange"
        conversation_id = storage.record_exchange(
            self.database,
            None,
            "What happened today?",
            "The indexed sources do not contain enough information.",
            [],
            0.1,
            exchange_id,
            "documents",
            True,
        )
        web_result = {
            "answer": "A web-grounded answer. ⟦Web 1, Web⟧",
            "sources": [{
                "source": "Example",
                "citation_label": "Web 1",
                "source_url": "https://example.com/fact",
                "source_type": "web",
                "preview": "",
                "distance": None,
            }],
            "answer_basis": "web",
            "web_search_available": False,
            "insufficient_information": False,
            "timings": {"total_s": 0.2},
        }

        with patch.object(rag_engine, "search_web", return_value=web_result) as search:
            response = self.client.post(
                f"/conversations/{conversation_id}/exchanges/{exchange_id}/search-web"
            )

        self.assertEqual(response.status_code, 200)
        search.assert_called_once_with("What happened today?", chat_history=[])
        saved = storage.get_conversation(self.database, conversation_id)["exchanges"][0]
        self.assertEqual(saved["answer_basis"], "web")
        self.assertFalse(saved["web_search_available"])
        self.assertEqual(saved["sources"][0]["source_url"], "https://example.com/fact")

    def test_web_search_rejects_an_ineligible_exchange(self):
        conversation_id = self.seed("A grounded question")
        exchange_id = storage.get_conversation(
            self.database, conversation_id,
        )["exchanges"][0]["id"]

        response = self.client.post(
            f"/conversations/{conversation_id}/exchanges/{exchange_id}/search-web"
        )

        self.assertEqual(response.status_code, 409)

    def test_web_search_rejects_an_older_exchange_before_calling_gemini(self):
        exchange_id = "older-eligible"
        conversation_id = storage.record_exchange(
            self.database, None, "Old question", "Insufficient", [], 0.1,
            exchange_id, "documents", True,
        )
        self.seed("Newer question", conversation_id)

        with patch.object(rag_engine, "search_web") as search:
            response = self.client.post(
                f"/conversations/{conversation_id}/exchanges/{exchange_id}/search-web"
            )

        self.assertEqual(response.status_code, 409)
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
