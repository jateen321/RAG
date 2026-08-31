"""Offline integration coverage for document identity across ingestion routes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from chromadb.config import Settings
from fastapi.testclient import TestClient

import api
import document_ingester
import indexer
from auth import AuthenticatedUser, get_current_user


class DocumentIdentityTests(unittest.TestCase):
    OWNER_ID = api.SHARED_CORPUS_OWNER_ID

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "data"
        self.root.mkdir()
        client = chromadb.PersistentClient(
            path=str(Path(self.temp.name) / "db"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection("test_documents")
        self.enterContext(patch.object(indexer, "_get_collection", return_value=self.collection))
        self.embed = self.enterContext(patch.object(
            indexer, "_embed_batch", side_effect=lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
        ))
        self.enterContext(patch.object(indexer, "_pace_delay", 0))
        self.enterContext(patch.object(api, "DATA_DIR", str(self.root)))
        api.app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            uid=self.OWNER_ID, email="test@example.com", is_admin=True
        )
        self.addCleanup(api.app.dependency_overrides.clear)
        self.client = TestClient(api.app)

    @staticmethod
    def pages(text="Evidence about the history of a book. " * 4):
        return [{"page": 1, "text": text, "method": "text"}]

    def test_folder_then_direct_and_reverse_for_all_supported_formats(self):
        for extension, mime in ((".pdf", "application/pdf"),
                                (".txt", "text/plain"), (".md", "text/markdown")):
            for folder_first in (True, False):
                with self.subTest(extension=extension, folder_first=folder_first):
                    name = f"lesson-{folder_first}{extension}"
                    folder = self.root / f"folder-{extension[1:]}-{folder_first}"
                    folder.mkdir()
                    nested = folder / name
                    body = (f"Unique evidence for {extension} {folder_first}. " * 4).encode()
                    nested.write_bytes(body)
                    pages = self.pages(body.decode())
                    before = self.collection.count()
                    with patch.object(api, "_extract_document", return_value=pages):
                        def upload():
                            return self.client.post("/upload", files={"file": (name, body, mime)})

                        def ingest_folder():
                            return document_ingester.index_folder(
                                folder, extract=lambda _: pages, owner_id=self.OWNER_ID
                            )

                        if folder_first:
                            self.assertEqual(ingest_folder()["files_indexed"], 1)
                            ids = set(self.collection.get()["ids"])
                            self.assertEqual(upload().status_code, 409)
                            self.assertFalse((self.root / name).exists())
                        else:
                            self.assertEqual(upload().status_code, 201)
                            ids = set(self.collection.get()["ids"])
                            self.assertEqual(ingest_folder()["files_skipped"], 1)
                    self.assertEqual(set(self.collection.get()["ids"]), ids)
                    self.assertEqual(self.collection.count(), before + 1)

    def test_same_filename_different_content_is_not_skipped(self):
        for folder_name, text in (("one", "First unrelated book. " * 5),
                                  ("two", "Second distinct book. " * 5)):
            folder = self.root / folder_name
            folder.mkdir()
            (folder / "notes.txt").write_text(text)
            report = document_ingester.index_folder(folder, owner_id=self.OWNER_ID)
            self.assertEqual(report["files_indexed"], 1)
        self.assertEqual(self.collection.count(), 2)
        documents = {md["document_id"] for md in self.collection.get(include=["metadatas"])["metadatas"]}
        self.assertEqual(len(documents), 2)

    def test_identical_files_in_one_folder_are_skipped_during_same_run(self):
        for name in ("first.txt", "renamed.txt"):
            (self.root / name).write_text("The same text under two different filenames. " * 5)
        report = document_ingester.index_folder(self.root, owner_id=self.OWNER_ID)
        self.assertEqual(report["files_indexed"], 1)
        self.assertEqual(report["files_skipped"], 1)
        self.assertEqual(self.embed.call_count, 1)

    def test_index_endpoint_reuses_folder_document_and_preserves_nested_source(self):
        folder = self.root / "book"
        folder.mkdir()
        path = folder / "notes.txt"
        path.write_text(self.pages()[0]["text"])
        response = self.client.post("/index", json={"filename": "book/notes.txt"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "book/notes.txt")
        count = self.collection.count()
        report = document_ingester.index_folder(
            folder, force=True, owner_id=self.OWNER_ID
        )
        self.assertEqual(report["files_skipped"], 1)
        response = self.client.post("/index", json={"filename": "book/notes.txt"})
        self.assertTrue(response.json()["deduplicated"])
        self.assertEqual(self.collection.count(), count)
        self.assertEqual(self.embed.call_count, 1)

    def test_legacy_document_is_reused_without_changing_citation_ids(self):
        pages = self.pages()
        chunks = indexer._chunk_text(pages[0]["text"], 1)
        indexer.index_chunks(chunks, "old/books/notes.txt", "text", document_key="/old/books/notes.txt")
        ids = self.collection.get()["ids"]
        path = self.root / "renamed.txt"
        path.write_text(pages[0]["text"])
        self.embed.reset_mock()
        self.assertEqual(indexer.index_document(pages, "renamed.txt", "text", file_path=path), 0)
        self.assertEqual(self.collection.get()["ids"], ids)
        self.assertTrue(indexer.is_document_indexed("renamed.txt", file_path=path))
        md = self.collection.get(include=["metadatas"])["metadatas"][0]
        self.assertEqual(md["source_name"], "old/books/notes.txt")
        self.embed.assert_not_called()

    def test_partial_index_is_resumed_through_another_path(self):
        pages = self.pages("A" * 80) + [{"page": 2, "text": "B" * 80, "method": "text"}]
        original = self.root / "original.pdf"
        copy = self.root / "copy.pdf"
        original.write_bytes(b"same PDF bytes")
        copy.write_bytes(original.read_bytes())
        with patch.object(indexer, "EMBED_BATCH_SIZE", 1):
            self.embed.side_effect = [[[0.1, 0.2, 0.3]], RuntimeError("quota")]
            with self.assertRaisesRegex(RuntimeError, "quota"):
                indexer.index_document(pages, "original.pdf", file_path=original)
            self.assertFalse(indexer.is_document_indexed("copy.pdf", file_path=copy))
            self.embed.reset_mock()
            self.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
            self.assertEqual(indexer.index_document(pages, "copy.pdf", file_path=copy), 2)
        self.assertEqual(self.embed.call_count, 1)
        self.assertTrue(indexer.is_document_indexed("copy.pdf", file_path=copy))
        self.assertEqual(self.collection.count(), 2)
        self.assertEqual({md["source_name"] for md in self.collection.get(include=["metadatas"])["metadatas"]}, {"original.pdf"})

    def test_extracted_content_identity_without_a_file_path(self):
        self.assertGreater(indexer.index_document(self.pages(), "notes.txt", "text"), 0)
        self.assertEqual(indexer.index_document(self.pages(), "folder/notes.txt", "text", document_key="/elsewhere/notes.txt"), 0)
        self.assertEqual(self.embed.call_count, 1)

    def test_shared_passage_does_not_merge_different_legacy_documents(self):
        first = self.pages("Shared introduction. " * 4)
        first.append({"page": 2, "text": "Different conclusion. " * 4})
        chunks = [c for page in first for c in indexer._chunk_text(page["text"], page["page"])]
        indexer.index_chunks(chunks, "folder/book.txt", "text")
        other = self.pages("Shared introduction. " * 4)
        other.append({"page": 2, "text": "A separate conclusion. " * 4})
        self.assertEqual(indexer.index_document(other, "book.txt", "text"), 2)
        self.assertEqual(self.collection.count(), 4)

    def test_reextraction_updates_metadata_on_reused_chunks(self):
        path = self.root / "book.pdf"
        path.write_bytes(b"same input document")
        pages = self.pages("A" * 80) + [{"page": 2, "text": "B" * 80}]
        indexer.index_document(pages, "book.pdf", file_path=path)
        self.embed.reset_mock()
        indexer.index_document(pages[:1], "copy.pdf", file_path=path)
        self.embed.assert_not_called()
        self.assertEqual(self.collection.count(), 1)
        self.assertTrue(indexer.is_document_indexed("book.pdf", file_path=path))
        self.assertEqual(self.collection.get(include=["metadatas"])["metadatas"][0]["chunk_total"], 1)

    def test_simultaneous_routes_store_only_one_document(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda source: indexer.index_document(self.pages(), source, "text"),
                ["book.txt", "folder/book.txt"],
            ))
        self.assertEqual(sorted(results), [0, 1])
        self.assertEqual(self.collection.count(), 1)
        self.assertEqual(self.embed.call_count, 1)

    def test_ocr_cache_rejects_same_name_size_and_mtime_with_different_content(self):
        import os
        first = self.root / "notes.pdf"
        second_dir = self.root / "nested"
        second_dir.mkdir()
        second = second_dir / first.name
        first.write_bytes(b"first")
        second.write_bytes(b"other")
        stat = first.stat()
        os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(document_ingester._cache_fingerprint(first),
                            document_ingester._cache_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
