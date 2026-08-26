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
    EMBED_BATCH_SIZE, EMBED_BATCH_DELAY_S, EMBED_MAX_ATTEMPTS,
    EMBED_BACKOFF_BASE_S,
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


def _chroma_metadata(values: dict) -> dict:
    """Return metadata values Chroma can store.

    Chroma accepts scalar values only. Dropping ``None`` and stringifying any
    unexpected value here keeps source adapters from leaking lists/dicts into
    the database and failing after the expensive embedding step.
    """
    scalar = (str, int, float, bool)
    return {
        key: value if isinstance(value, scalar) else str(value)
        for key, value in values.items()
        if value is not None
    }


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


def _is_quota_error(exc: Exception) -> bool:
    """True for a rate-limit/quota rejection, whatever shape the SDK gives it."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "quota" in text or "resource_exhausted" in text


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed one batch, retrying a quota rejection with exponential backoff.

    The previous version retried exactly once after a fixed 30s, and — crucially
    — the retry call itself was NOT inside a try block, so a second 429 escaped
    as a raw traceback and abandoned the whole document. Quota here is a
    per-minute window, so a single short wait is not reliably enough once that
    window has been saturated.
    """
    delay = EMBED_BACKOFF_BASE_S
    for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
        try:
            result = _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
            return [e.values for e in result.embeddings]
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            if attempt == EMBED_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Embedding quota still exhausted after {EMBED_MAX_ATTEMPTS} "
                    f"attempts. Nothing was written for this document — rerun "
                    f"the index once the per-minute quota has recovered."
                ) from exc
            console.print(
                f"[yellow]⏳ Rate limited (attempt {attempt}/"
                f"{EMBED_MAX_ATTEMPTS}) — waiting {delay}s...[/yellow]"
            )
            time.sleep(delay)
            delay *= 2


def _embed_texts(texts: list[str], batch_size: int = None, on_progress=None) -> list[list[float]]:
    """
    Embed texts with the configured Gemini embedding model.

    Batches, paces, and retries. The pacing sleep here previously never ran:
    ``index_document`` pre-sliced the work and called this with
    ``batch_size=len(batch)``, so the loop always executed exactly one iteration
    and the "is there another batch?" guard was never true. Callers now hand the
    FULL list over and receive progress through ``on_progress`` instead of
    re-implementing the batching.

    Args:
        texts: All chunk texts to embed.
        batch_size: Chunks per request. Defaults to ``EMBED_BATCH_SIZE``.
        on_progress: Optional callable invoked with the count just completed.
    """
    batch_size = batch_size or EMBED_BATCH_SIZE
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(_embed_batch(batch))

        if on_progress:
            on_progress(len(batch))

        # Proactive spacing between batches. Cheaper than being throttled:
        # a 1s pause costs ~1 minute over 72 batches, while one 429 costs a
        # 10-80s backoff and risks losing the document entirely.
        if i + batch_size < len(texts):
            time.sleep(EMBED_BATCH_DELAY_S)

    return all_embeddings


def _get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(
    chunks: list[dict],
    source_name: str,
    source_type: str,
    *,
    document_key: str | None = None,
    source_metadata: dict | None = None,
) -> int:
    """Embed and store already-created chunks from any source adapter.

    Every chunk must contain ``text`` and ``chunk_index``. Location-specific
    fields (for example ``page_number`` or ``start_seconds``) are stored as
    metadata and passed through retrieval. ``document_key`` is the stable
    identity of the source; YouTube uses a video ID so title changes replace
    the same document instead of creating a duplicate.
    """
    if not chunks:
        console.print("[red]❌ No text chunks created.[/red]")
        return 0

    console.print(f"   Created [bold]{len(chunks)}[/bold] chunks")
    console.print("\n[bold]🧠 Generating embeddings...[/bold]")
    texts = [chunk["text"] for chunk in chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks", total=len(texts))
        embeddings = _embed_texts(
            texts, on_progress=lambda n: progress.update(task, advance=n)
        )

    console.print("\n[bold]💾 Storing in vector database...[/bold]")
    collection = _get_collection()
    before = collection.count()
    doc_id = _document_id(document_key or source_name, source_type)
    common_metadata = _chroma_metadata(source_metadata or {})

    ids = [
        (
            f"{doc_id}_p{chunk['page_number']:04d}_c{chunk['chunk_index']:03d}"
            if chunk.get("page_number") is not None
            else f"{doc_id}_c{chunk['chunk_index']:05d}"
        )
        for chunk in chunks
    ]
    for i in range(0, len(chunks), 100):
        batch_chunks = chunks[i:i + 100]
        collection.upsert(
            ids=ids[i:i + 100],
            embeddings=embeddings[i:i + 100],
            documents=[chunk["text"] for chunk in batch_chunks],
            metadatas=[
                _chroma_metadata({
                    **common_metadata,
                    **{k: v for k, v in chunk.items() if k != "text"},
                    "source_type": source_type,
                    "document_id": doc_id,
                    "source_name": source_name,
                })
                for chunk in batch_chunks
            ],
        )

    # Upsert replaces current positions but cannot remove positions left over
    # from an older, longer extraction. Remove those only after all new chunks
    # are safely written; an embedding failure therefore leaves the old source
    # untouched.
    stored_ids = collection.get(where={"document_id": doc_id}, include=[])["ids"]
    current_ids = set(ids)
    stale_ids = [stored_id for stored_id in stored_ids if stored_id not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    total_in_db = collection.count()
    added = total_in_db - before
    console.print(f"\n[green]✅ Indexed {len(chunks)} chunks![/green]")
    console.print(f"   📊 Total chunks in database: {total_in_db}")
    if added != len(chunks):
        console.print(
            f"   [dim]ℹ️  {max(added, 0)} new; existing chunks for this "
            "document were replaced.[/dim]"
        )
    return len(chunks)


def index_document(
    pages_text: list[dict], source_name: str, source_type: str = "pdf"
) -> int:
    """
    Index extracted text into ChromaDB.

    Args:
        pages_text: List of {'page': int, 'text': str, 'method': str} from the
            OCR engine. 'method' is 'direct' or 'ocr'.
        source_name: Name of the source file, e.g. "CIL.pdf" or "notes.md".
        source_type: Kind of source this text came from, such as "pdf", "text",
            "markdown", or "youtube".

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

    return index_chunks(all_chunks, source_name, source_type)


def _find_document_id(source_name: str) -> str | None:
    """Resolve a displayed source name to its stored document ID."""
    metadatas = _get_collection().get(include=["metadatas"])["metadatas"] or []
    matches = {
        md.get("document_id")
        for md in metadatas
        if md and md.get("source_name", "").casefold() == source_name.casefold()
    }
    matches.discard(None)
    return next(iter(matches)) if len(matches) == 1 else None


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
    doc_id = _find_document_id(source_name)
    if not doc_id:
        return 0

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
    doc_id = _find_document_id(source_name)
    if not doc_id:
        return []
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
            "source_type": md.get("source_type", "unknown"),
            "start_seconds": md.get("start_seconds"),
            "end_seconds": md.get("end_seconds"),
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
        types_by_source: dict[str, str] = {}
        for md in metadatas:
            md = md or {}
            source = md.get("source_name", "unknown")
            types_by_source[source] = md.get("source_type", "unknown")
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
                "source_type": types_by_source.get(source, "unknown"),
            })

        return {
            "total_chunks": count,
            "db_path": CHROMA_DB_PATH,
            "documents": documents,
        }
    except Exception:
        return {"total_chunks": 0, "db_path": CHROMA_DB_PATH, "documents": []}
