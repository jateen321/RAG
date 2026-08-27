import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import indexer
import rag_engine
import retriever


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

    def test_legacy_citation_resolves_only_one_exact_preview_match(self):
        preview = "The complete cited paragraph."
        collection = Mock()
        collection.get.return_value = {
            "ids": ["doc_p0002_c003", "doc_p0002_c004"],
            "documents": [preview, "A different paragraph."],
            "metadatas": [
                {
                    "source_name": "book.pdf",
                    "source_type": "pdf",
                    "page_number": 2,
                    "chunk_index": 3,
                },
                {
                    "source_name": "book.pdf",
                    "source_type": "pdf",
                    "page_number": 2,
                    "chunk_index": 4,
                },
            ],
        }
        with (
            patch.object(indexer, "_find_document_id", return_value="doc"),
            patch.object(indexer, "_get_collection", return_value=collection),
        ):
            passage = indexer.get_chunk_by_citation("book.pdf", 2, preview)

        self.assertEqual(passage["chunk_id"], "doc_p0002_c003")
        self.assertEqual(passage["text"], preview)
        collection.get.assert_called_once_with(
            where={
                "$and": [
                    {"document_id": "doc"},
                    {"page_number": 2},
                ]
            },
            include=["documents", "metadatas"],
        )

    def test_legacy_citation_refuses_ambiguous_preview_matches(self):
        preview = "Repeated text"
        collection = Mock()
        collection.get.return_value = {
            "ids": ["first", "second"],
            "documents": [preview, preview],
            "metadatas": [
                {"source_name": "book.pdf", "page_number": 2},
                {"source_name": "book.pdf", "page_number": 2},
            ],
        }
        with (
            patch.object(indexer, "_find_document_id", return_value="doc"),
            patch.object(indexer, "_get_collection", return_value=collection),
        ):
            passage = indexer.get_chunk_by_citation("book.pdf", 2, preview)

        self.assertIsNone(passage)


class CitationIdentityTests(unittest.TestCase):
    def test_retriever_exposes_the_stable_chroma_chunk_id(self):
        embedding_client = Mock()
        embedding_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2])]
        )
        collection = Mock()
        collection.count.return_value = 1
        collection.query.return_value = {
            "ids": [["doc_p0002_c003"]],
            "documents": [["The complete cited paragraph."]],
            "metadatas": [[{
                "source_name": "book.pdf",
                "source_type": "pdf",
                "page_number": 2,
            }]],
            "distances": [[0.12]],
        }
        chroma_client = Mock()
        chroma_client.get_collection.return_value = collection

        with (
            patch.object(retriever, "_client", embedding_client),
            patch.object(
                retriever.chromadb,
                "PersistentClient",
                return_value=chroma_client,
            ),
        ):
            result = retriever.retrieve("question", top_k=1)

        self.assertEqual(result[0]["chunk_id"], "doc_p0002_c003")

    def test_rag_response_preserves_the_retrieved_chunk_id(self):
        chunk = {
            "chunk_id": "doc_p0002_c003",
            "text": "The complete cited paragraph.",
            "page": 2,
            "source": "book.pdf",
            "distance": 0.12,
            "source_type": "pdf",
        }
        model_client = Mock()
        with (
            patch.object(rag_engine, "retrieve", return_value=[chunk]),
            patch.object(rag_engine, "_client", model_client),
            patch.object(rag_engine, "_answer_text", return_value="Answer"),
        ):
            result = rag_engine.ask_with_sources("question", top_k=1)

        self.assertEqual(result["sources"][0]["chunk_id"], "doc_p0002_c003")


class PromptLanguageContractTests(unittest.TestCase):
    def test_english_grammar_is_not_overridden_by_a_transliterated_term(self):
        prompt = rag_engine.SYSTEM_PROMPT

        self.assertIn("question alone", prompt)
        self.assertIn('"Hi, what is Bhagya?" must receive an English answer', prompt)
        self.assertIn("Do not let the language of the passages", prompt)

    def test_romanized_hindi_and_explicit_overrides_are_covered(self):
        prompt = rag_engine.SYSTEM_PROMPT

        self.assertIn('"Bhagya kya hai?" must receive a Hindi answer', prompt)
        self.assertIn("explicitly requests a response language", prompt)


if __name__ == "__main__":
    unittest.main()
