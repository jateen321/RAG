"""
Indexer — Chunks text and stores embeddings in ChromaDB.

Takes OCR-extracted text, splits into chunks, embeds with Gemini,
and stores in a persistent ChromaDB collection.
"""

import re
import time
import hashlib
import os
import unicodedata
import chromadb
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH, CHROMA_DB_PATH, COLLECTION_NAME,
)
from llm_client import get_client

console = Console()

# Initialize Gemini client (backend chosen in config: Developer API or Vertex)
_client = get_client()


# Sentence-ending marks across our scripts:
#   . ! ?   → English      ।  ॥   → Devanagari danda / double danda
_SENTENCE_END = re.compile(r'(?<=[.!?।॥])\s+')

# Zero-width characters OCR sprinkles in (invisible, but they split tokens and
# make otherwise-identical words compare unequal): ZWSP, ZWNJ, ZWJ, BOM.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def _clean_text(text: str) -> str:
    """Normalize extracted/OCR text before chunking.

    - NFC: compose Devanagari matras into their canonical single form so the
      same word always has the same codepoints (better matching + embedding).
    - Drop zero-width joiners: OCR noise, invisible, but they fragment tokens.
    Blank lines (paragraph breaks) are preserved for _split_units; per-sentence
    whitespace is collapsed later, so no info for splitting is lost here.
    """
    text = unicodedata.normalize("NFC", text)
    return text.translate(_ZERO_WIDTH)


def _split_units(text: str) -> list[str]:
    """Break text into the smallest pieces we refuse to cut across:
    paragraphs first (blank line), then sentences within each paragraph.
    Internal whitespace (incl. mid-sentence line breaks) is collapsed so a
    unit reads as one clean line."""
    units = []
    for para in re.split(r'\n\s*\n', text):
        for sent in _SENTENCE_END.split(para.strip()):
            sent = re.sub(r'\s+', ' ', sent).strip()
            if sent:
                units.append(sent)
    return units


def _document_id(source_name: str, source_type: str = "pdf") -> str:
    """Stable identifier for a document, independent of filename casing.

    macOS filesystems are case-insensitive, so `os.path.basename()` records
    whatever the user *typed* ("bhagya-...") rather than what is on disk
    ("Bhagya-..."). Hashing the casefolded name means both spellings resolve to
    the same document, so a re-index updates the document instead of silently
    creating a second one under a different capitalization.
    """
    key = f"{source_type}:{source_name.casefold()}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _content_hash(text: str) -> str:
    """Fingerprint of a chunk's text. Two chunks with the same hash hold
    identical content — useful for spotting duplicates and for telling whether
    a re-extraction (e.g. a DPI change) actually altered the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _chunk_text(text: str, page_num: int) -> list[dict]:
    """
    Split text into overlapping chunks on natural boundaries.

    Greedily packs whole sentences up to CHUNK_SIZE so a chunk never ends
    mid-word or mid-sentence. Consecutive chunks overlap by whole trailing
    sentences, carried back until at least CHUNK_OVERLAP characters are reached
    (so a tiny marker like "(2)" can't become a useless overlap) but never past
    MAX_CHUNK_OVERLAP (so one long sentence can't be duplicated wholesale into
    the next chunk). A single unit longer than CHUNK_SIZE (e.g. an OCR page with
    almost no punctuation) is hard-split on character size as a fallback.

    Returns list of dicts with 'text', 'page_number', 'chunk_index' and
    'content_hash'. The chunk's id is assigned by ``index_document``, which is
    the only layer that knows the document identity.
    """
    # 0. Normalize the text (NFC + strip zero-width OCR noise).
    text = _clean_text(text)

    # 1. Sentence-sized units, hard-splitting any unit bigger than a chunk.
    units = []
    for u in _split_units(text):
        if len(u) > CHUNK_SIZE:
            units += [u[i:i + CHUNK_SIZE] for i in range(0, len(u), CHUNK_SIZE)]
        else:
            units.append(u)

    # 2. Greedily pack units into chunks up to CHUNK_SIZE.
    chunks = []
    current = []          # units accumulated in the chunk being built
    current_len = 0
    chunk_index = 0

    def flush():
        nonlocal chunk_index
        chunk = " ".join(current).strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": chunk,
                "page_number": page_num,
                "chunk_index": chunk_index,
                "content_hash": _content_hash(chunk),
            })
            chunk_index += 1

    for unit in units:
        # +1 accounts for the space we join sentences with.
        if current and current_len + len(unit) + 1 > CHUNK_SIZE:
            flush()
            # 3. Carry whole trailing sentences back as the next chunk's head:
            #    enough to bridge context (>= CHUNK_OVERLAP), but never so much
            #    that the next chunk is largely a copy of this one
            #    (<= MAX_CHUNK_OVERLAP). The ceiling is checked BEFORE accepting
            #    a sentence, so one long sentence can no longer be carried whole.
            #    Iterating current[1:] leaves the first unit behind, so two
            #    consecutive chunks can never be fully nested.
            overlap, olen = [], 0
            for prev in reversed(current[1:]):
                if olen + len(prev) + 1 > MAX_CHUNK_OVERLAP:
                    break          # ceiling wins over the CHUNK_OVERLAP floor
                overlap.insert(0, prev)
                olen += len(prev) + 1
                if olen >= CHUNK_OVERLAP:
                    break          # bridge is long enough
            current, current_len = overlap, olen
        current.append(unit)
        current_len += len(unit) + 1

    flush()  # the final chunk

    return chunks


def _embed_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """
    Embed texts using Gemini's free embedding model.

    Handles rate limiting by batching and adding small delays.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            result = _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                console.print("[yellow]⏳ Rate limited, waiting 30 seconds...[/yellow]")
                time.sleep(30)
                # Retry this batch
                result = _client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                all_embeddings.extend([e.values for e in result.embeddings])
            else:
                raise e

        # Small delay to respect free tier rate limits
        if i + batch_size < len(texts):
            time.sleep(1)

    return all_embeddings


def _get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_document(
    pages_text: list[dict], source_name: str, source_type: str = "pdf"
) -> int:
    """
    Index extracted text into ChromaDB.

    Args:
        pages_text: List of {'page': int, 'text': str, 'method': str} from the
            OCR engine. 'method' is 'direct' or 'ocr'.
        source_name: Name of the source file, e.g. "CIL.pdf".
        source_type: Kind of source this text came from. Only "pdf" is produced
            today; the field exists so a future transcript/web ingester can
            share the same collection without its chunks being indistinguishable
            from PDF pages.

    Returns:
        Number of chunks indexed.
    """
    # Step 1: Chunk all pages
    console.print("\n[bold]📦 Chunking text...[/bold]")
    all_chunks = []
    for page_data in pages_text:
        chunks = _chunk_text(page_data["text"], page_data["page"])
        # ocr_engine records how each page was read ('direct' vs 'ocr'). Carry
        # it onto every chunk from that page so retrieval failures can later be
        # correlated with the extraction method that produced them.
        method = page_data.get("method", "unknown")
        for chunk in chunks:
            chunk["extraction_method"] = method
        all_chunks.extend(chunks)

    if not all_chunks:
        console.print("[red]❌ No text chunks created. The PDF might be empty.[/red]")
        return 0

    console.print(f"   Created [bold]{len(all_chunks)}[/bold] chunks from {len(pages_text)} pages")

    # Step 2: Embed chunks
    console.print("\n[bold]🧠 Generating embeddings (free Gemini API)...[/bold]")
    texts = [c["text"] for c in all_chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks", total=len(texts))
        embeddings = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = _embed_texts(batch, batch_size=len(batch))
            embeddings.extend(batch_embeddings)
            progress.update(task, advance=len(batch))

    # Step 3: Store in ChromaDB
    console.print("\n[bold]💾 Storing in vector database...[/bold]")
    collection = _get_collection()
    before = collection.count()

    doc_id = _document_id(source_name, source_type)

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        # The id is derived from document + position, NOT from the chunk text.
        # Hashing the text meant that re-extracting a page (a DPI change, an OCR
        # fix) produced brand-new ids, so the old chunks were left behind and the
        # index accumulated two generations of the same page. Position-derived
        # ids + upsert make a re-index replace in place.
        ids = [
            f"{doc_id}_p{c['page_number']:04d}_c{c['chunk_index']:03d}"
            for c in batch_chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{
                "source_type": source_type,
                "document_id": doc_id,
                "source_name": source_name,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "extraction_method": c["extraction_method"],
                "content_hash": c["content_hash"],
            } for c in batch_chunks],
        )

    total_in_db = collection.count()
    added = total_in_db - before
    console.print(f"\n[green]✅ Indexed {len(all_chunks)} chunks![/green]")
    console.print(f"   📊 Total chunks in database: {total_in_db}")

    # Report what actually landed, not just what we tried to write. A re-index
    # that replaces existing chunks legitimately adds 0 — but silence here is
    # what let a no-op masquerade as success.
    if added != len(all_chunks):
        console.print(
            f"   [dim]ℹ️  {added} new, {len(all_chunks) - added} replaced "
            f"existing chunks for this document.[/dim]"
        )

    return len(all_chunks)


def remove_document(source_name: str) -> int:
    """Delete every chunk that came from one source document.

    Chunks carry a "document_id" metadata field, so Chroma can delete them with
    a metadata filter — no need to know the individual chunk ids. Lets you drop
    a single document without a full ``reset``.

    Matching on document_id rather than source_name makes removal
    case-insensitive: "bhagya-x.pdf" and "Bhagya-x.pdf" resolve to the same
    document, which is exactly the mismatch that previously let a re-index
    create a second copy under different capitalization.

    Args:
        source_name: The stored source name, e.g. "CIL.pdf" (as shown by `status`).

    Returns:
        Number of chunks deleted (0 if that source isn't indexed).
    """
    collection = _get_collection()
    doc_id = _document_id(source_name)

    # Count first: delete(where=...) doesn't report how much it removed.
    matching = collection.get(where={"document_id": doc_id}, include=[])["ids"]
    if not matching:
        return 0

    collection.delete(where={"document_id": doc_id})
    return len(matching)


def get_document_chunks(source_name: str) -> list[dict]:
    """Every stored chunk for one document, in reading order.

    Matches on document_id so casing doesn't matter, and sorts by
    (page_number, chunk_index) — Chroma returns rows in arbitrary order, but
    consecutive chunks only make sense to look at in sequence, since the whole
    point of inspecting them is to see how they overlap and where they break.

    Args:
        source_name: Source name as shown by `status`, e.g. "CIL.pdf".

    Returns:
        List of {'text', 'page_number', 'chunk_index', 'extraction_method',
        'content_hash', 'chunk_id'}; empty if that document isn't indexed.
    """
    collection = _get_collection()
    doc_id = _document_id(source_name)
    got = collection.get(
        where={"document_id": doc_id}, include=["documents", "metadatas"]
    )

    rows = []
    for cid, text, md in zip(got["ids"], got["documents"], got["metadatas"]):
        md = md or {}
        rows.append({
            "chunk_id": cid,
            "text": text,
            "page_number": md.get("page_number", 0),
            "chunk_index": md.get("chunk_index", 0),
            "extraction_method": md.get("extraction_method", "unknown"),
            "content_hash": md.get("content_hash", ""),
        })

    rows.sort(key=lambda r: (r["page_number"], r["chunk_index"]))
    return rows


def get_stats() -> dict:
    """Get statistics about the indexed documents.

    Every chunk carries {"page", "source"} metadata, so we can aggregate it into
    a per-document breakdown. Only metadata is fetched — pulling documents or
    embeddings back would be needlessly expensive just to count things.

    Returns:
        {
          "total_chunks": int,
          "db_path": str,
          "documents": [{"source", "chunks", "pages", "first_page", "last_page"}]
        }
        ``documents`` is sorted by source name and is empty when nothing is indexed.
    """
    try:
        collection = _get_collection()
        count = collection.count()
        if not count:
            return {"total_chunks": 0, "db_path": CHROMA_DB_PATH, "documents": []}

        metadatas = collection.get(include=["metadatas"])["metadatas"] or []

        # source -> set of pages it contributed (a page yields several chunks)
        pages_by_source: dict[str, set] = {}
        chunks_by_source: dict[str, int] = {}
        methods_by_source: dict[str, dict[str, int]] = {}
        for md in metadatas:
            md = md or {}
            source = md.get("source_name", "unknown")
            chunks_by_source[source] = chunks_by_source.get(source, 0) + 1
            page = md.get("page_number")
            if page is not None:
                pages_by_source.setdefault(source, set()).add(page)
            method = md.get("extraction_method", "unknown")
            methods_by_source.setdefault(source, {})
            methods_by_source[source][method] = (
                methods_by_source[source].get(method, 0) + 1
            )

        documents = []
        for source in sorted(chunks_by_source):
            pages = sorted(pages_by_source.get(source, set()))
            documents.append({
                "source": source,
                "chunks": chunks_by_source[source],
                "pages": len(pages),
                "first_page": pages[0] if pages else None,
                "last_page": pages[-1] if pages else None,
                "methods": methods_by_source.get(source, {}),
            })

        return {
            "total_chunks": count,
            "db_path": CHROMA_DB_PATH,
            "documents": documents,
        }
    except Exception:
        return {"total_chunks": 0, "db_path": CHROMA_DB_PATH, "documents": []}
