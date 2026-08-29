import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class DocumentUploadValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

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
        get_chunk.assert_called_once_with("doc_p0002_c003", "book.pdf")

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
            "book.pdf", 2, "The complete cited paragraph."
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

                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json()["source"], filename)
                index_document.assert_called_once_with(
                    pages, filename, source_type, file_path=Path(data_dir).resolve() / filename
                )

    def test_existing_unindexed_pdf_is_indexed_without_overwrite(self):
        pages = [{"page": 1, "text": "Original content", "method": "text"}]
        with tempfile.TemporaryDirectory() as data_dir:
            existing = Path(data_dir) / "lesson.pdf"
            existing.write_bytes(b"original content")

            with (
                patch.object(api, "DATA_DIR", data_dir),
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
                pages, "lesson.pdf", "pdf", file_path=existing.resolve()
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

            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["source"], "psychology/lesson.pdf")
            self.assertEqual(
                (Path(data_dir) / "psychology" / "lesson.pdf").read_bytes(),
                b"content",
            )
            index_document.assert_called_once_with(
                pages, "psychology/lesson.pdf", "pdf",
                file_path=Path(data_dir).resolve() / "psychology" / "lesson.pdf",
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
        index_folder.assert_called_once_with(Path(allowed).resolve(), True)


class ConversationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "conversations.sqlite3"
        self.database_patch = patch.object(
            api, "CONVERSATION_DB_PATH", str(self.database_path)
        )
        self.database_patch.start()

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
