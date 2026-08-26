"""
RAG Engine — Orchestrates retrieval and generation.

Retrieves relevant chunks, builds a context-aware prompt,
and generates answers using the configured Gemini model.
"""

import time
from typing import NoReturn

from google.genai import types
from google.genai.errors import ClientError
from rich.console import Console
from rich.markup import escape

from config import LLM_MODEL
from llm_client import get_client
from retriever import retrieve

console = Console()

# Initialize Gemini client (backend chosen in config: Developer API or Vertex)
_client = get_client()

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful study assistant. Your job is to help students learn and understand indexed PDFs and YouTube transcripts.

RULES:
1. Answer ONLY based on the provided context from indexed sources.
2. If the context doesn't contain enough information, say so honestly.
3. Reply in the SAME LANGUAGE as the student's question (Hindi or English).
4. Cite PDF page number(s) or YouTube timestamp(s), matching the provided labels.
5. Explain concepts clearly, as if teaching a student.
6. If asked to summarize, provide a clear and concise summary.
7. Use bullet points and formatting to make answers easy to read."""


def _quota_guard(exc: ClientError) -> NoReturn:
    """Translate a Gemini 429 (quota/rate limit) into a RuntimeError, which the
    API layer maps to a 503. Any other ClientError propagates unchanged."""
    if exc.code == 429:
        raise RuntimeError(
            "Gemini quota/rate limit reached. Please try again later."
        ) from exc
    raise exc


def _build_user_message(chunks: list[dict], question: str) -> str:
    """Build the grounding prompt. Shared by ``ask`` and ``ask_with_sources``.

    Every passage is labelled with its SOURCE FILE as well as its page. Without
    the filename the model receives several interchangeable "Page N" blocks that
    may come from different books, so the page citations rule 4 asks for are
    ambiguous across the corpus, and facts from unrelated documents can be
    blended into one answer as though they shared a source.

    This lives in one place because the two callers previously built the same
    prompt independently and had already drifted apart.
    """
    def label(chunk: dict) -> str:
        if chunk.get("source_type") == "youtube":
            timestamp = chunk.get("timestamp") or "0:00"
            return f"{chunk['source']} · Timestamp {timestamp}"
        return f"{chunk['source']} · पृष्ठ {chunk['page']} / Page {chunk['page']}"

    context = "\n\n---\n\n".join(
        f"[{label(chunk)}]:\n{chunk['text']}" for chunk in chunks
    )
    return f"""Context from indexed sources:

{context}

---

Student's question: {question}

Please answer based on the context above."""


def _answer_text(response) -> str:
    """Pull the text out of a Gemini response, refusing to return an empty one.

    ``response.text`` is None when the model produced no usable candidate — a
    safety block, a recitation stop, an empty finish. Returning that silently
    hands None to the CLI and the API as though it were an answer.
    """
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError(
            "Gemini returned no answer text (the response was empty, blocked, "
            "or stopped early)."
        )
    return text


def ask(question: str, chat_history: list = None, show_sources: bool = True) -> str:
    """
    Answer a question using RAG (Retrieve + Generate).

    Args:
        question: User's question in Hindi or English.
        chat_history: Optional list of previous messages for context.
        show_sources: Whether to display source chunks.

    Returns:
        The generated answer string.

    Raises:
        RuntimeError: on a quota/rate limit, or when the model returned no
            usable text. Other ClientErrors propagate unchanged. Callers must
            handle these — they are no longer folded into the returned string.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question)

    if not chunks:
        return "❌ कोई प्रासंगिक जानकारी नहीं मिली। कृपया पहले एक PDF इंडेक्स करें।\n(No relevant information found. Please index a PDF first.)"

    # Step 2: Show sources if requested
    if show_sources:
        console.print("\n[dim]📚 Sources found:[/dim]")
        for chunk in chunks:
            source = chunk["source"]
            # escape(): the preview is raw OCR text, and console.print reads
            # `[...]` as style markup — an unescaped bracket would be eaten
            # (or raise MarkupError) instead of being shown.
            preview = escape(chunk["text"][:80].replace("\n", " ")) + "..."
            location = (
                f"Timestamp {chunk.get('timestamp') or '0:00'}"
                if chunk.get("source_type") == "youtube"
                else f"पृष्ठ/Page {chunk['page']}"
            )
            console.print(
                f"   • [cyan]{escape(source)}[/cyan] "
                f"[dim]{location}:[/dim] [dim]{preview}[/dim]"
            )
        console.print()

    # Step 3: Build the prompt (shared with ask_with_sources)
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append(
        {"role": "user", "parts": [{"text": _build_user_message(chunks, question)}]}
    )

    # Step 4: Generate answer with Gemini (new SDK).
    # Failures are RAISED, not returned as answer text: a caller cannot tell a
    # real answer from an error string, and the evaluation harness would happily
    # score an error message as a response. This now matches ask_with_sources,
    # which already raised.
    try:
        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    except ClientError as e:
        _quota_guard(e)                  # 429 → RuntimeError; else re-raised

    return _answer_text(response)


def ask_with_sources(question: str, top_k: int = None) -> dict:
    """
    Answer a question and return the answer, its sources, and phase timings.

    Programmatic sibling of ``ask`` (no rich console output). Used by the
    FastAPI ``/ask`` route and the evaluation harness.

    ``top_k`` is exposed so the evaluation harness can retrieve once and score
    several cut-offs from the same ranked list. Previously the harness called
    ``retrieve`` itself AND called this function, which retrieved a second time
    — doubling the embedding calls per question against a quota that is already
    the binding constraint on a full run.

    ``timings`` separates retrieval from generation so latency can be attributed
    rather than reported as one opaque end-to-end number.

    Args:
        question: The user's question.
        top_k: Chunks to retrieve. Defaults to ``config.TOP_K``.

    Returns:
        {"answer": str,
         "sources": [{"page", "source", "distance", "preview", ...}],
         "timings": {"retrieval_s", "generation_s", "total_s"}}

    Raises:
        RuntimeError: if nothing is indexed yet (empty / missing collection),
            so the API can surface a 503.
    """
    started = time.perf_counter()
    try:
        kwargs = {"top_k": top_k} if top_k is not None else {}
        chunks = retrieve(question, **kwargs)   # embedding call — can raise 429
    except ClientError as e:
        _quota_guard(e)                  # 429 → RuntimeError; else re-raised
    retrieved_at = time.perf_counter()

    if not chunks:
        raise RuntimeError(
            "No indexed documents found. Index a PDF before asking questions."
        )

    try:
        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": _build_user_message(chunks, question)}]}
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    except ClientError as e:
        _quota_guard(e)                  # 429 → RuntimeError; else re-raised

    answer = _answer_text(response)
    finished = time.perf_counter()

    sources = [
        {
            "page": c["page"],
            "source": c["source"],
            "distance": c["distance"],
            "extraction_method": c.get("extraction_method", "unknown"),
            "preview": c["text"][:120].replace("\n", " "),
            "source_type": c.get("source_type", "unknown"),
            "timestamp": c.get("timestamp"),
            "start_seconds": c.get("start_seconds"),
            "end_seconds": c.get("end_seconds"),
            "source_url": c.get("source_url"),
            "video_id": c.get("video_id"),
            "video_title": c.get("video_title"),
            "channel_name": c.get("channel_name"),
            "channel_id": c.get("channel_id"),
            "duration_seconds": c.get("duration_seconds"),
            "upload_date": c.get("upload_date"),
            "transcript_language": c.get("transcript_language"),
            "transcript_language_code": c.get("transcript_language_code"),
            "transcript_is_generated": c.get("transcript_is_generated"),
            "playlist_id": c.get("playlist_id"),
            "playlist_title": c.get("playlist_title"),
            "playlist_index": c.get("playlist_index"),
            "playlist_url": c.get("playlist_url"),
        }
        for c in chunks
    ]
    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "retrieval_s": round(retrieved_at - started, 3),
            "generation_s": round(finished - retrieved_at, 3),
            "total_s": round(finished - started, 3),
        },
    }
