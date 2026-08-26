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
                    pages, filename, source_type
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
            index_document.assert_called_once_with(pages, "lesson.pdf", "pdf")

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


if __name__ == "__main__":
    unittest.main()
