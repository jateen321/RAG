"""
Indexer — Chunks text and stores embeddings in ChromaDB.

Takes OCR-extracted text, splits into chunks, embeds with Gemini,
and stores in a persistent ChromaDB collection.
"""

import time
import hashlib
import os
from google import genai
import chromadb
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH, CHROMA_DB_PATH, COLLECTION_NAME,
)

console = Console()

# Initialize Gemini client (new SDK)
_client = genai.Client(api_key=GEMINI_API_KEY)


def _chunk_text(text: str, page_num: int) -> list[dict]:
    """
    Split text into overlapping chunks.

    Returns list of dicts with 'text', 'page', and 'chunk_id'.
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_LENGTH:
            # Create a unique ID from content hash
            chunk_id = hashlib.md5(
                f"p{page_num}_c{chunk_index}_{chunk[:50]}".encode()
            ).hexdigest()

            chunks.append({
                "text": chunk,
                "page": page_num,
                "chunk_id": chunk_id,
            })
            chunk_index += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

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


def index_document(pages_text: list[dict], source_name: str) -> int:
    """
    Index extracted text into ChromaDB.

    Args:
        pages_text: List of {'page': int, 'text': str} from OCR engine.
        source_name: Name of the source PDF file.

    Returns:
        Number of chunks indexed.
    """
    # Step 1: Chunk all pages
    console.print("\n[bold]📦 Chunking text...[/bold]")
    all_chunks = []
    for page_data in pages_text:
        chunks = _chunk_text(page_data["text"], page_data["page"])
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

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{
                "page": c["page"],
                "source": source_name,
            } for c in batch_chunks],
        )

    total_in_db = collection.count()
    console.print(f"\n[green]✅ Indexed {len(all_chunks)} chunks![/green]")
    console.print(f"   📊 Total chunks in database: {total_in_db}")

    return len(all_chunks)


def get_stats() -> dict:
    """Get statistics about the indexed documents."""
    try:
        collection = _get_collection()
        count = collection.count()
        return {"total_chunks": count, "db_path": CHROMA_DB_PATH}
    except Exception:
        return {"total_chunks": 0, "db_path": CHROMA_DB_PATH}
