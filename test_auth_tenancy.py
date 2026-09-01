import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import api
import auth
import conversation_store as storage
import retriever


class AllowAllLimiter:
    def __init__(self):
        self.identities = []

    @asynccontextmanager
    async def admit(self, identity, *, rates=(), concurrency=()):
        self.identities.append(identity)
        yield


class FirebaseAuthenticationTests(unittest.TestCase):
    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_protected_route_rejects_missing_session(self):
        response = TestClient(api.app).get("/conversations")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Sign in to continue.")

    def test_verified_claims_become_application_identity(self):
        firebase = Mock()
        firebase.verify_session_cookie.return_value = {
            "uid": "student-123",
            "email": "student@example.com",
            "admin": True,
        }
        with patch.object(auth, "_firebase_auth", return_value=firebase):
            user = auth.verify_session_cookie("signed-cookie")

        self.assertEqual(user.uid, "student-123")
        self.assertEqual(user.email, "student@example.com")
        self.assertTrue(user.is_admin)
        firebase.verify_session_cookie.assert_called_once_with(
            "signed-cookie", check_revoked=auth.AUTH_CHECK_REVOKED
        )

    def test_session_token_failure_is_logged_without_exposing_it_to_client(self):
        firebase = Mock()
        firebase.verify_id_token.side_effect = ValueError("Firebase rejected token")

        with (
            patch.object(auth, "_firebase_auth", return_value=firebase),
            self.assertLogs("auth", level="WARNING") as captured,
            self.assertRaises(api.HTTPException) as raised,
        ):
            auth.create_session_cookie("private-id-token", 3600)

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(
            raised.exception.detail, "Google sign-in could not be verified."
        )
        self.assertIn("ValueError", captured.output[0])
        self.assertIn("Firebase rejected token", captured.output[0])
        self.assertNotIn("private-id-token", captured.output[0])

    def test_untrusted_origin_cannot_create_session(self):
        response = TestClient(api.app).post(
            "/auth/session",
            headers={"Origin": "https://attacker.example"},
            json={"id_token": "x" * 30},
        )
        self.assertEqual(response.status_code, 403)

    def test_http_development_cookie_authenticates_after_page_refresh(self):
        user = auth.AuthenticatedUser(
            uid="student-123", email="student@example.com"
        )
        client = TestClient(api.app, base_url="http://localhost")

        with (
            patch.object(api, "SESSION_COOKIE_SECURE", False),
            patch.object(api, "create_session_cookie", return_value="signed-cookie"),
            patch.object(api, "verify_session_cookie", return_value=user),
        ):
            login = client.post(
                "/auth/session",
                headers={"Origin": "http://localhost:3000"},
                json={"id_token": "x" * 30},
            )

        with patch.object(auth, "verify_session_cookie", return_value=user) as verify:
            refreshed = client.get("/auth/me")

        self.assertEqual(login.status_code, 200)
        self.assertNotIn("; Secure", login.headers["set-cookie"])
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.json()["uid"], "student-123")
        verify.assert_called_once_with("signed-cookie")

    def test_untrusted_origin_cannot_reach_state_changing_routes(self):
        # A valid session is deliberately supplied: the forged origin, not a
        # missing cookie, must be what rejects the request.
        api.app.dependency_overrides[auth.get_current_user] = (
            lambda: auth.AuthenticatedUser(uid="student-123")
        )
        client = TestClient(api.app)
        forged = {"Origin": "https://attacker.example"}

        with patch("rag_engine.ask_with_sources") as ask:
            asked = client.post("/ask", json={"question": "Leak it"}, headers=forged)
        deleted = client.delete("/conversations/any-id", headers=forged)

        self.assertEqual(asked.status_code, 403)
        self.assertEqual(asked.json()["detail"], "Untrusted request origin.")
        self.assertEqual(deleted.status_code, 403)
        ask.assert_not_called()

    def test_origin_check_does_not_block_reads(self):
        response = TestClient(api.app).get(
            "/conversations", headers={"Origin": "https://attacker.example"}
        )
        # Reads stay guarded by the session cookie alone, so top-level PDF and
        # generated-image links keep working without an Origin header.
        self.assertEqual(response.status_code, 401)

    def test_invalid_session_cookie_does_not_downgrade_to_guest(self):
        request = Mock(cookies={auth.SESSION_COOKIE_NAME: "expired-cookie"})
        with patch.object(
            auth,
            "verify_session_cookie",
            side_effect=api.HTTPException(status_code=401, detail="expired"),
        ):
            with self.assertRaises(api.HTTPException) as raised:
                auth.get_optional_user(request)

        self.assertEqual(raised.exception.status_code, 401)

    def test_guest_question_uses_shared_corpus_without_persistence(self):
        answer = {
            "answer": "Shared answer ⟦Jateen_Resume.pdf, Page 1⟧",
            "sources": [],
            "timings": {"total_s": 0.1},
        }
        limiter = AllowAllLimiter()
        with (
            patch.object(api, "get_rate_limiter", return_value=limiter),
            patch("rag_engine.ask_with_sources", return_value=answer) as ask,
            patch.object(api, "record_exchange") as record,
        ):
            response = TestClient(api.app).post(
                "/ask", json={"question": "What is in the shared corpus?"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["conversation_id"])
        self.assertEqual(response.json()["answer"], "Shared answer")
        self.assertNotIn("sources", response.json())
        ask.assert_called_once_with(
            "What is in the shared corpus?",
            chat_history=[],
            owner_id=api.SHARED_CORPUS_OWNER_ID,
        )
        record.assert_not_called()
        self.assertTrue(limiter.identities[0].startswith("guest:"))

    def test_guest_cannot_request_authenticated_modes(self):
        limiter = AllowAllLimiter()
        with (
            patch.object(api, "get_rate_limiter", return_value=limiter),
            patch("rag_engine.search_web") as search,
        ):
            response = TestClient(api.app).post(
                "/ask",
                json={"question": "Search this", "use_web": True},
            )

        self.assertEqual(response.status_code, 401)
        search.assert_not_called()

    def test_non_admin_cannot_upload_to_shared_corpus(self):
        api.app.dependency_overrides[auth.get_current_user] = lambda: (
            auth.AuthenticatedUser(uid="ordinary-user", is_admin=False)
        )
        response = TestClient(api.app).post(
            "/upload",
            files={"file": ("notes.txt", b"private", "text/plain")},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Administrator access is required.")


class ConversationTenancyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "conversations.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_users_cannot_list_read_continue_or_delete_each_others_conversations(self):
        conversation_id = storage.record_exchange(
            self.database,
            None,
            "Private question",
            "Private answer",
            [],
            0.1,
            owner_id="user-a",
        )

        self.assertEqual(
            storage.list_conversations(self.database, owner_id="user-b"), []
        )
        self.assertIsNone(
            storage.get_conversation(self.database, conversation_id, "user-b")
        )
        self.assertFalse(
            storage.conversation_exists(self.database, conversation_id, "user-b")
        )
        with self.assertRaisesRegex(ValueError, "Conversation not found"):
            storage.record_exchange(
                self.database,
                conversation_id,
                "Intruding follow-up",
                "Should not save",
                [],
                0.1,
                owner_id="user-b",
            )
        self.assertFalse(
            storage.delete_conversation(self.database, conversation_id, "user-b")
        )
        self.assertIsNotNone(
            storage.get_conversation(self.database, conversation_id, "user-a")
        )

    def test_generated_image_metadata_is_owner_scoped(self):
        conversation_id = storage.record_exchange(
            self.database, None, "Visual", "Answer", [], 0.1,
            exchange_id="exchange-a", image_generation_available=True,
            owner_id="user-a",
        )
        self.assertTrue(storage.attach_generated_image(
            self.database, conversation_id, "exchange-a", "image-a", "image/png",
            owner_id="user-a",
        ))
        self.assertIsNone(storage.get_generated_image_metadata(
            self.database, "image-a", owner_id="user-b",
        ))


class DocumentTenancyTests(unittest.TestCase):
    def test_legacy_migration_assigns_only_unowned_vectors_to_shared_corpus(self):
        collection = Mock()
        collection.get.return_value = {
            "ids": ["legacy", "private"],
            "metadatas": [{"source_name": "shared.pdf"}, {"owner_id": "user-a"}],
        }
        with patch("indexer._get_collection", return_value=collection):
            from indexer import assign_legacy_documents

            updated = assign_legacy_documents(api.SHARED_CORPUS_OWNER_ID)

        self.assertEqual(updated, 1)
        collection.update.assert_called_once_with(
            ids=["legacy"],
            metadatas=[{
                "source_name": "shared.pdf",
                "owner_id": api.SHARED_CORPUS_OWNER_ID,
            }],
        )

    def test_legacy_migration_batches_large_collections(self):
        collection = Mock()
        collection.get.return_value = {
            "ids": ["one", "two", "three"],
            "metadatas": [{}, {}, {}],
        }
        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer.OWNER_MIGRATION_BATCH_SIZE", 2),
        ):
            from indexer import assign_legacy_documents

            updated = assign_legacy_documents(api.SHARED_CORPUS_OWNER_ID)

        self.assertEqual(updated, 3)
        self.assertEqual(collection.update.call_count, 2)
        self.assertEqual(collection.update.call_args_list[0].kwargs["ids"], ["one", "two"])
        self.assertEqual(collection.update.call_args_list[1].kwargs["ids"], ["three"])

    def test_one_users_document_path_does_not_resolve_for_another_user(self):
        with (
            tempfile.TemporaryDirectory() as data_dir,
            patch.object(api, "DATA_DIR", data_dir),
            patch.object(api, "FIREBASE_PROJECT_ID", "configured-project"),
        ):
            owner_a_root = api._tenant_data_root("user-a")
            owner_a_root.mkdir(parents=True)
            (owner_a_root / "private.txt").write_text("private", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Document not found"):
                api._resolve_data_document("private.txt", "user-b")

    def test_vector_query_is_filtered_by_owner(self):
        collection = Mock()
        collection.get.return_value = {"ids": ["owned-chunk"]}
        collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]],
        }
        client = Mock()
        client.get_collection.return_value = collection
        embedding = Mock(values=[0.1, 0.2])
        with (
            patch.object(retriever.chromadb, "PersistentClient", return_value=client),
            patch.object(
                retriever._client.models,
                "embed_content",
                return_value=Mock(embeddings=[embedding]),
            ),
        ):
            retriever.retrieve_many(["question"], owner_id="user-a")

        collection.query.assert_called_once()
        self.assertEqual(
            collection.query.call_args.kwargs["where"], {"owner_id": "user-a"}
        )


if __name__ == "__main__":
    unittest.main()
