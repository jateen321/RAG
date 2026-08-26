import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import document_ingester


class FolderIngestionTests(unittest.TestCase):
    def test_discovers_supported_files_recursively_in_stable_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "nested").mkdir()
            (root / "b.TXT").write_text("b", encoding="utf-8")
            (root / "a.md").write_text("a", encoding="utf-8")
            (root / "nested" / "c.pdf").write_bytes(b"pdf")
            (root / "ignored.docx").write_bytes(b"no")

            found = document_ingester.discover_documents(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in found],
                ["a.md", "b.TXT", "nested/c.pdf"],
            )

    def test_non_recursive_discovery_excludes_nested_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "nested").mkdir()
            (root / "top.txt").write_text("top", encoding="utf-8")
            (root / "nested" / "deep.txt").write_text("deep", encoding="utf-8")

            found = document_ingester.discover_documents(root, recursive=False)

            self.assertEqual([path.name for path in found], ["top.txt"])

    def test_allowed_folder_rejects_paths_outside_roots(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as other:
            with self.assertRaisesRegex(ValueError, "outside INDEX_FOLDER_ROOTS"):
                document_ingester.resolve_allowed_folder(other, [allowed])

    def test_folder_report_keeps_relative_metadata_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "nested").mkdir()
            good = root / "nested" / "good.md"
            bad = root / "bad.txt"
            good.write_text("good", encoding="utf-8")
            bad.write_text("bad", encoding="utf-8")

            def extract(path):
                if path == bad:
                    raise ValueError("broken encoding")
                return [{"page": 1, "text": "content", "method": "text"}]

            with patch("indexer.index_document", return_value=3) as index_document:
                report = document_ingester.index_folder(root, extract=extract)

            self.assertEqual(report["files_found"], 2)
            self.assertEqual(report["files_indexed"], 1)
            self.assertEqual(report["files_failed"], 1)
            self.assertEqual(report["chunks_indexed"], 3)
            kwargs = index_document.call_args.kwargs
            self.assertEqual(kwargs["source_metadata"]["relative_path"], "nested/good.md")
            self.assertEqual(kwargs["source_metadata"]["file_extension"], ".md")


if __name__ == "__main__":
    unittest.main()
