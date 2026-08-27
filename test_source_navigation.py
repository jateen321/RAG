import unittest
from unittest.mock import Mock, patch

import indexer


class ExactChunkLookupTests(unittest.TestCase):
    def test_get_chunk_returns_text_only_for_the_requested_source(self):
        collection = Mock()
        collection.get.return_value = {
            "ids": ["doc_p0002_c003"],
            "documents": ["The complete cited paragraph."],
            "metadatas": [
                {
                    "source_name": "book.pdf",
                    "source_type": "pdf",
                    "page_number": 2,
                    "chunk_index": 3,
                }
            ],
        }
        with patch.object(indexer, "_get_collection", return_value=collection):
            passage = indexer.get_chunk("doc_p0002_c003", "BOOK.PDF")
            mismatch = indexer.get_chunk("doc_p0002_c003", "other.pdf")

        self.assertEqual(passage["text"], "The complete cited paragraph.")
        self.assertEqual(passage["page_number"], 2)
        self.assertIsNone(mismatch)

    def test_get_chunk_returns_none_for_an_unknown_id(self):
        collection = Mock()
        collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        with patch.object(indexer, "_get_collection", return_value=collection):
            self.assertIsNone(indexer.get_chunk("missing", "book.pdf"))


if __name__ == "__main__":
    unittest.main()
