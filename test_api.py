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

    def test_existing_pdf_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as data_dir, patch.object(api, "DATA_DIR", data_dir):
            existing = Path(data_dir) / "lesson.pdf"
            existing.write_bytes(b"original content")

            response = self.client.post(
                "/upload",
                files={"file": ("lesson.pdf", b"replacement", "application/pdf")},
            )

            self.assertEqual(existing.read_bytes(), b"original content")

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
