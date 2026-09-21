import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

from chroma_lock import ChromaStoreError, validate_chroma_store


class ChromaPreflightTests(unittest.TestCase):
    def _store(self, dimension=3, metadata_dimension=3):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        connection = sqlite3.connect(root / "chroma.sqlite3")
        connection.executescript(
            """
            CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER);
            CREATE TABLE segments (
                id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO collections VALUES ('collection', 'hindi_textbook', ?)",
            (dimension,),
        )
        connection.execute(
            "INSERT INTO segments VALUES ('segment', 'vector', 'VECTOR', 'collection')"
        )
        connection.execute(
            "INSERT INTO segments VALUES ('metadata', 'metadata', 'METADATA', 'collection')"
        )
        connection.execute("INSERT INTO embeddings VALUES (1, 'metadata', 'chunk')")
        connection.commit()
        connection.close()
        segment = root / "segment"
        segment.mkdir()
        (segment / "index_metadata.pickle").write_bytes(
            pickle.dumps({"dimensionality": metadata_dimension})
        )
        self.addCleanup(temp.cleanup)
        return root

    def test_new_store_without_sqlite_is_allowed(self):
        with tempfile.TemporaryDirectory() as path:
            validate_chroma_store(path)

    def test_valid_segment_passes(self):
        validate_chroma_store(self._store())

    def test_missing_metadata_fails_closed(self):
        root = self._store()
        (root / "segment" / "index_metadata.pickle").unlink()
        with self.assertRaisesRegex(ChromaStoreError, "no index_metadata"):
            validate_chroma_store(root)

    def test_invalid_dimension_fails_closed(self):
        root = self._store(metadata_dimension=None)
        with self.assertRaisesRegex(ChromaStoreError, "invalid dimensionality"):
            validate_chroma_store(root)

    def test_dimension_mismatch_fails_closed(self):
        root = self._store(dimension=3072, metadata_dimension=768)
        with self.assertRaisesRegex(ChromaStoreError, "dimension mismatch"):
            validate_chroma_store(root)


if __name__ == "__main__":
    unittest.main()
