import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from auth import AuthenticatedUser, get_current_user, get_optional_user


TEST_USER = AuthenticatedUser(
    uid="firebase-test-user",
    email="student@example.com",
    is_admin=True,
)


class DocumentUploadValidationTests(unittest.TestCase):
    def setUp(self):
        api.app.dependency_overrides[get_current_user] = lambda: TEST_USER
        api.app.dependency_overrides[get_optional_user] = lambda: TEST_USER
        self.client = TestClient(api.app)

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_rejects_unsupported_upload(self):
        with tempfile.TemporaryDirectory() as data_dir, patch.object(api, "DATA_DIR", data_dir):
            response = self.client.post(
                "/upload",
                files={"file": ("notes.docx", b"not supported", "application/octet-stream")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Only PDF, TXT, and Markdown files can be uploaded.",
        )

    def test_extracts_utf8_text_and_markdown(self):
        with tempfile.TemporaryDirectory() as data_dir:
            for filename, content in (
                ("notes.txt", "Interview notes"),
                ("guide.md", "# Study guide"),
            ):
                path = Path(data_dir) / filename
                path.write_text(content, encoding="utf-8")
                self.assertEqual(
                    api._extract_text_document(path),
                    [{"page": 1, "text": content, "method": "text"}],
                )

    def test_document_route_opens_a_nested_text_source(self):
        with tempfile.TemporaryDirectory() as data_dir:
            document = Path(data_dir) / "Mahabharata" / "maha09.txt"
            document.parent.mkdir()
            document.write_text("Mahabharata passage", encoding="utf-8")
            with patch.object(api, "DATA_DIR", data_dir):
                response = self.client.get("/documents/Mahabharata/maha09.txt")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Mahabharata passage")
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))

    def test_guests_cannot_read_shared_sources_or_passages(self):
        api.app.dependency_overrides.clear()
        guest = TestClient(api.app)

        document = guest.get("/documents/Mahabharata/maha09.txt")
        passage = guest.get(
            "/passages/doc_p0002_c003", params={"source": "book.pdf"}
        )

        self.assertEqual(document.status_code, 401)
        self.assertEqual(passage.status_code, 401)

    def test_guest_health_is_liveness_only(self):
        api.app.dependency_overrides.clear()
        with patch("indexer.get_stats") as stats:
            response = TestClient(api.app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        stats.assert_not_called()

    def test_document_route_accepts_legacy_data_prefix(self):
        with tempfile.TemporaryDirectory() as data_dir:
            document = Path(data_dir) / "essence-of-hinduism.pdf"
            document.write_bytes(b"%PDF-1.4\n")
            with patch.object(api, "DATA_DIR", data_dir):
                response = self.client.get("/documents/data/essence-of-hinduism.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/pdf"))

    def test_passage_route_returns_the_exact_indexed_chunk(self):
        passage = {
            "chunk_id": "doc_p0002_c003",
            "source": "book.pdf",
            "text": "The complete cited paragraph.",
            "page_number": 2,
            "chunk_index": 3,
            "source_type": "pdf",
        }
        with patch("indexer.get_chunk", return_value=passage) as get_chunk:
            response = self.client.get(
                "/passages/doc_p0002_c003", params={"source": "book.pdf"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), passage)
        get_chunk.assert_called_once_with(
            "doc_p0002_c003", "book.pdf", api.SHARED_CORPUS_OWNER_ID
        )

    def test_passage_route_hides_missing_or_mismatched_chunks(self):
        with patch("indexer.get_chunk", return_value=None):
            response = self.client.get(
                "/passages/private_chunk", params={"source": "another.pdf"}
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Passage not found.")

    def test_legacy_passage_route_resolves_an_exact_citation(self):
        passage = {
            "chunk_id": "doc_p0002_c003",
            "source": "book.pdf",
            "text": "The complete cited paragraph.",
            "page_number": 2,
            "chunk_index": 3,
            "source_type": "pdf",
        }
        with patch(
            "indexer.get_chunk_by_citation", return_value=passage
        ) as resolve:
            response = self.client.get(
                "/passages/resolve-legacy",
                params={
                    "source": "book.pdf",
                    "page": 2,
                    "preview": "The complete cited paragraph.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), passage)
        resolve.assert_called_once_with(
            "book.pdf", 2, "The complete cited paragraph.",
            api.SHARED_CORPUS_OWNER_ID,
        )

    def test_rejects_non_utf8_text(self):
        with tempfile.TemporaryDirectory() as data_dir:
            path = Path(data_dir) / "notes.txt"
            path.write_bytes(b"\xff\xfe\x00")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                api._extract_text_document(path)

    def test_uploads_text_and_markdown_with_correct_source_type(self):
        pages = [{"page": 1, "text": "Study notes", "method": "text"}]
        for filename, content_type, source_type in (
            ("notes.txt", "text/plain", "text"),
            ("guide.md", "text/markdown", "markdown"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as data_dir:
                with (
                    patch.object(api, "DATA_DIR", data_dir),
                    patch.object(api, "_extract_document", return_value=pages),
                    patch("indexer.is_document_indexed", return_value=False),
                    patch("indexer.index_document", return_value=2) as index_document,
                ):
                    response = self.client.post(
                        "/upload",
                        files={"file": (filename, b"Study notes", content_type)},
                    )
                    shared_file = (
                        api._tenant_data_root(api.SHARED_CORPUS_OWNER_ID) / filename
                    )

                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json()["source"], filename)
                index_document.assert_called_once_with(
                    pages, filename, source_type,
                    file_path=shared_file,
                    owner_id=api.SHARED_CORPUS_OWNER_ID,
                )

    def test_existing_unindexed_pdf_is_indexed_without_overwrite(self):
        pages = [{"page": 1, "text": "Original content", "method": "text"}]
        with (
            tempfile.TemporaryDirectory() as data_dir,
            patch.object(api, "DATA_DIR", data_dir),
        ):
            existing = api._tenant_data_root(api.SHARED_CORPUS_OWNER_ID) / "lesson.pdf"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"original content")

            with (
                patch.object(api, "_extract_document", return_value=pages),
                patch("indexer.is_document_indexed", return_value=False),
                patch("indexer.index_document", return_value=2) as index_document,
            ):
                response = self.client.post(
                    "/upload",
                    files={"file": ("lesson.pdf", b"replacement", "application/pdf")},
                )

            self.assertEqual(existing.read_bytes(), b"original content")
            index_document.assert_called_once_with(
                pages, "lesson.pdf", "pdf", file_path=existing.resolve(),
                owner_id=api.SHARED_CORPUS_OWNER_ID,
            )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["used_existing_file"])

    def test_rejects_document_already_indexed_in_chromadb(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with (
                patch.object(api, "DATA_DIR", data_dir),
                patch("indexer.is_document_indexed", return_value=True),
            ):
                response = self.client.post(
                    "/upload",
                    files={"file": ("lesson.pdf", b"content", "application/pdf")},
                )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "'lesson.pdf' is already indexed in the library.",
        )

    def test_upload_preserves_a_folder_relative_path(self):
        pages = [{"page": 1, "text": "Folder notes", "method": "text"}]
        with tempfile.TemporaryDirectory() as data_dir:
            with (
                patch.object(api, "DATA_DIR", data_dir),
                patch.object(api, "_extract_document", return_value=pages),
                patch("indexer.is_document_indexed", return_value=False),
                patch("indexer.index_document", return_value=1) as index_document,
            ):
                response = self.client.post(
                    "/upload",
                    data={"relative_path": "psychology/lesson.pdf"},
                    files={"file": ("lesson.pdf", b"content", "application/pdf")},
                )
                shared_file = (
                    api._tenant_data_root(api.SHARED_CORPUS_OWNER_ID)
                    / "psychology" / "lesson.pdf"
                )

            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["source"], "psychology/lesson.pdf")
            self.assertEqual(
                shared_file.read_bytes(), b"content",
            )
            index_document.assert_called_once_with(
                pages, "psychology/lesson.pdf", "pdf",
                file_path=shared_file,
                owner_id=api.SHARED_CORPUS_OWNER_ID,
            )

    def test_upload_rejects_relative_path_traversal(self):
        with tempfile.TemporaryDirectory() as data_dir, patch.object(api, "DATA_DIR", data_dir):
            response = self.client.post(
                "/upload",
                data={"relative_path": "../lesson.pdf"},
                files={"file": ("lesson.pdf", b"content", "application/pdf")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid document path.")

    def test_folder_endpoint_rejects_a_path_outside_allowlist(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as other:
            with patch.object(api, "INDEX_FOLDER_ROOTS", [allowed]):
                response = self.client.post(
                    "/index/folder",
                    json={"folder_path": other, "recursive": True},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("outside INDEX_FOLDER_ROOTS", response.json()["detail"])

    def test_folder_endpoint_indexes_an_allowlisted_path(self):
        report = {
            "folder_path": "/library",
            "recursive": True,
            "files_found": 2,
            "files_indexed": 2,
            "files_skipped": 0,
            "files_failed": 0,
            "chunks_indexed": 4,
            "results": [],
        }
        with tempfile.TemporaryDirectory() as allowed:
            with (
                patch.object(api, "INDEX_FOLDER_ROOTS", [allowed]),
                patch.object(api, "index_folder", return_value=report) as index_folder,
            ):
                response = self.client.post(
                    "/index/folder",
                    json={"folder_path": allowed, "recursive": True},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), report)
        index_folder.assert_called_once_with(
            Path(allowed).resolve(), True, owner_id=api.SHARED_CORPUS_OWNER_ID
        )


class ConversationHistoryTests(unittest.TestCase):
    def setUp(self):
        api.app.dependency_overrides[get_current_user] = lambda: TEST_USER
        api.app.dependency_overrides[get_optional_user] = lambda: TEST_USER
        self.client = TestClient(api.app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "conversations.sqlite3"
        self.database_patch = patch.object(
            api, "CONVERSATION_DB_PATH", str(self.database_path)
        )
        self.database_patch.start()

    def tearDown(self):
        api.app.dependency_overrides.clear()
        self.database_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def answer(answer: str = "Kolkata is in West Bengal.") -> dict:
        return {
            "answer": answer,
            "sources": [
                {
                    "source": "geography.pdf",
                    "page": 4,
                    "distance": 0.12,
                    "preview": "Kolkata is the capital of West Bengal.",
                }
            ],
            "timings": {"total_s": 1.25},
        }

    def test_successful_answers_are_persisted_and_loaded(self):
        with patch("rag_engine.ask_with_sources", return_value=self.answer()):
            response = self.client.post(
                "/ask", json={"question": "Where is Kolkata?"}
            )

        self.assertEqual(response.status_code, 200)
        conversation_id = response.json()["conversation_id"]

        history = self.client.get("/conversations")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()["conversations"]), 1)
        self.assertEqual(
            history.json()["conversations"][0]["title"], "Where is Kolkata?"
        )

        detail = self.client.get(f"/conversations/{conversation_id}")
        self.assertEqual(detail.status_code, 200)
        exchange = detail.json()["exchanges"][0]
        self.assertEqual(exchange["question"], "Where is Kolkata?")
        self.assertEqual(exchange["answer"], "Kolkata is in West Bengal.")
        self.assertEqual(exchange["sources"][0]["source"], "geography.pdf")
        self.assertEqual(exchange["total_seconds"], 1.25)

    def test_image_prompt_is_forwarded_but_not_persisted(self):
        image_bytes = b"\x89PNG\r\n\x1a\nsmall payload"
        with patch("rag_engine.ask_with_sources", return_value=self.answer()) as ask:
            response = self.client.post(
                "/ask/image",
                data={"question": "Explain this diagram"},
                files={"image": ("diagram.png", image_bytes, "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        ask.assert_called_once_with(
            "Explain this diagram",
            chat_history=[],
            image_data=image_bytes,
            image_mime_type="image/png",
            owner_id=api.SHARED_CORPUS_OWNER_ID,
        )
        conversation_id = response.json()["conversation_id"]
        exchange = self.client.get(
            f"/conversations/{conversation_id}"
        ).json()["exchanges"][0]
        self.assertEqual(exchange["question"], "Explain this diagram")
        self.assertNotIn("image", exchange)

    def test_web_mode_uses_search_directly_when_selected(self):
        web_answer = {
            **self.answer("Current web answer."),
            "answer_basis": "web",
        }
        with (
            patch("rag_engine.search_web", return_value=web_answer) as search,
            patch("rag_engine.ask_with_sources") as documents,
        ):
            response = self.client.post(
                "/ask",
                json={"question": "What happened today?", "use_web": True},
            )

        self.assertEqual(response.status_code, 200)
        search.assert_called_once_with("What happened today?", chat_history=[])
        documents.assert_not_called()
        self.assertEqual(response.json()["answer_basis"], "web")

    def test_image_mode_generates_and_attaches_image_when_selected(self):
        generated = {"image_data": b"generated", "image_mime_type": "image/png"}
        grounded_prompt = "Student request: Explain photosynthesis\nRetrieved evidence: chlorophyll"
        answer = {**self.answer(), "image_prompt": grounded_prompt}
        image_directory = Path(self.temp_dir.name) / "generated-images"
        with (
            patch.object(api, "GENERATED_IMAGE_DIR", image_directory),
            patch("rag_engine.ask_with_sources", return_value=answer) as ask,
            patch("rag_engine.generate_image", return_value=generated) as generate,
        ):
            response = self.client.post(
                "/ask",
                json={"question": "Explain photosynthesis", "generate_image": True},
            )

        self.assertEqual(response.status_code, 200)
        ask.assert_called_once_with(
            "Explain photosynthesis",
            chat_history=[],
            prepare_image_prompt=True,
            owner_id=api.SHARED_CORPUS_OWNER_ID,
        )
        generate.assert_called_once_with(grounded_prompt)
        payload = response.json()
        self.assertNotIn("image_prompt", payload)
        self.assertTrue(payload["generated_image_id"])
        self.assertTrue((image_directory / payload["generated_image_id"]).is_file())
        exchange = self.client.get(
            f"/conversations/{payload['conversation_id']}"
        ).json()["exchanges"][0]
        self.assertEqual(exchange["generated_image_id"], payload["generated_image_id"])

    def test_image_prompt_rejects_unsupported_type(self):
        with patch("rag_engine.ask_with_sources") as ask:
            response = self.client.post(
                "/ask/image",
                data={"question": "Explain this"},
                files={"image": ("vector.svg", b"<svg/>", "image/svg+xml")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"], "Choose a PNG, JPEG, or WebP image."
        )
        ask.assert_not_called()

    def test_image_prompt_rejects_mismatched_contents(self):
        with patch("rag_engine.ask_with_sources") as ask:
            response = self.client.post(
                "/ask/image",
                data={"question": "Explain this"},
                files={"image": ("fake.png", b"not a png", "image/png")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "The image contents do not match its file type.",
        )
        ask.assert_not_called()

    def test_image_prompt_rejects_oversized_file(self):
        with (
            patch.object(api, "MAX_PROMPT_IMAGE_BYTES", 8),
            patch("rag_engine.ask_with_sources") as ask,
        ):
            response = self.client.post(
                "/ask/image",
                data={"question": "Explain this"},
                files={
                    "image": (
                        "large.png",
                        b"\x89PNG\r\n\x1a\nextra",
                        "image/png",
                    )
                },
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json()["detail"], "Choose an image no larger than 10 MB."
        )
        ask.assert_not_called()

    def test_existing_conversation_accepts_more_exchanges_and_can_be_deleted(self):
        with patch("rag_engine.ask_with_sources", return_value=self.answer()):
            first = self.client.post("/ask", json={"question": "First question"})
            conversation_id = first.json()["conversation_id"]
            second = self.client.post(
                "/ask",
                json={
                    "question": "Follow-up question",
                    "conversation_id": conversation_id,
                },
            )

        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["conversation_id"], conversation_id)
        detail = self.client.get(f"/conversations/{conversation_id}")
        self.assertEqual(len(detail.json()["exchanges"]), 2)

        deleted = self.client.delete(f"/conversations/{conversation_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(
            self.client.get(f"/conversations/{conversation_id}").status_code, 404
        )

    def test_unknown_conversation_is_rejected_before_asking_gemini(self):
        with patch("rag_engine.ask_with_sources") as ask_with_sources:
            response = self.client.post(
                "/ask",
                json={"question": "Hello", "conversation_id": "missing"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Conversation not found.")
        ask_with_sources.assert_not_called()

    def test_editing_an_exchange_regenerates_from_that_point(self):
        answers = [self.answer("First answer"), self.answer("Second answer"), self.answer("Third answer")]
        with patch("rag_engine.ask_with_sources", side_effect=answers):
            first = self.client.post("/ask", json={"question": "First question"})
            conversation_id = first.json()["conversation_id"]
            self.client.post("/ask", json={"question": "Second question", "conversation_id": conversation_id})
            self.client.post("/ask", json={"question": "Third question", "conversation_id": conversation_id})

        before = self.client.get(f"/conversations/{conversation_id}").json()["exchanges"]
        edited_id = before[1]["id"]
        edited_answer = self.answer("Edited answer")
        with patch("rag_engine.ask_with_sources", return_value=edited_answer) as ask:
            response = self.client.put(
                f"/conversations/{conversation_id}/exchanges/{edited_id}",
                json={"question": "Edited second question"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["exchange_id"], edited_id)
        history = ask.call_args.kwargs["chat_history"]
        self.assertEqual([item["parts"][0]["text"] for item in history], ["First question", "First answer"])
        after = self.client.get(f"/conversations/{conversation_id}").json()["exchanges"]
        self.assertEqual(len(after), 2)
        self.assertEqual(after[1]["id"], edited_id)
        self.assertEqual(after[1]["question"], "Edited second question")
        self.assertEqual(after[1]["answer"], "Edited answer")

    def test_editing_first_exchange_updates_title(self):
        with patch("rag_engine.ask_with_sources", return_value=self.answer()):
            created = self.client.post("/ask", json={"question": "Original title"})
            conversation_id = created.json()["conversation_id"]
            exchange_id = created.json()["exchange_id"]
            response = self.client.put(
                f"/conversations/{conversation_id}/exchanges/{exchange_id}",
                json={"question": "Replacement title"},
            )

        self.assertEqual(response.status_code, 200)
        detail = self.client.get(f"/conversations/{conversation_id}").json()
        self.assertEqual(detail["title"], "Replacement title")

    def test_editing_unknown_exchange_is_rejected_before_generation(self):
        with patch("rag_engine.ask_with_sources", return_value=self.answer()):
            created = self.client.post("/ask", json={"question": "First question"})
        with patch("rag_engine.ask_with_sources") as ask:
            response = self.client.put(
                f"/conversations/{created.json()['conversation_id']}/exchanges/missing",
                json={"question": "Edited question"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Exchange not found.")
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
