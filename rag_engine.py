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

# System prompt for the LLM.
#
# Rule numbers are load-bearing: `evaluate.py::_is_refusal` and FINDINGS §8.5
# cite "rule 2" for refusal, and `_build_user_message` below cites "rule 4" for
# citations. Renumber these and those references silently become wrong.
SYSTEM_PROMPT = """You are a study assistant for a library of indexed documents and YouTube transcripts. Students ask in Hindi or English, and you answer from the passages retrieved for their question — only from those.

WHAT YOU ARE GIVEN
Each passage is labelled with its source and a locator, in one of three forms:
  [filename · पृष्ठ N / Page N]        — a page of a document
  [filename · Document section N]      — a section of a text file
  [video title · Timestamp M:SS]       — a moment in a transcript
Passages are selected by similarity alone, so some may be irrelevant to the question. About two-thirds are OCR'd from scanned books: expect broken words, wrong characters, and missing punctuation.

RULES:
1. Ground every claim in the passages above. Do not add facts from your own knowledge — not even about famous texts you recognise, and not even when you are certain you are right. The passages are the only authority here; if they do not say it, you do not know it.
2. If the passages do not answer the question, say so and stop. Open that reply with exactly "The provided context does not contain this information." or, in Hindi, "दिए गए संदर्भ में यह जानकारी उपलब्ध नहीं है।" A passage on the same topic is not an answer: being about Gandhi is not the same as stating when he died. Never close the gap by guessing.
3. Reply in the language of the question — a Hindi question gets a Hindi answer. Quote evidence in its original language, and translate it when it differs from the language you are answering in.
4. Cite a source for every claim, as "(source, locator)" — for example "(SRIMAD-BHAGAVAD-GITA.pdf, Page 301)", "(mahabharata.txt, Document section 12)", or "(Gita Lecture 3, Timestamp 14:05)". Copy the locator exactly as its label gives it. Never invent, adjust, or round one, and never cite a passage that is not listed above.
5. Use only the passages that bear on the question and ignore the rest; several unrelated books may be retrieved together. If two passages disagree, say so and cite both rather than silently picking one.
6. Read through OCR damage where the meaning is clear, and quote damaged text as it appears rather than repairing it. If a passage is too corrupted to read with confidence, say that instead of guessing at what it meant.
7. Teach, don't just answer: define terms the student may not know and show the reasoning behind the conclusion. Match the shape of the answer to the question — a factual question deserves a short direct reply; use bullet points, headings, or a summary structure only when the content genuinely calls for one."""


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
        if chunk.get("source_type") in {"text", "markdown"}:
            return f"{chunk['source']} · Document section {chunk['page']}"
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
        return "❌ कोई प्रासंगिक जानकारी नहीं मिली। कृपया पहले एक दस्तावेज़ इंडेक्स करें।\n(No relevant information found. Please index a document first.)"

    # Step 2: Show sources if requested
    if show_sources:
        console.print("\n[dim]📚 Sources found:[/dim]")
        for chunk in chunks:
            source = chunk["source"]
            # escape(): the preview is raw OCR text, and console.print reads
            # `[...]` as style markup — an unescaped bracket would be eaten
            # (or raise MarkupError) instead of being shown.
            preview = escape(chunk["text"][:80].replace("\n", " ")) + "..."
            if chunk.get("source_type") == "youtube":
                location = f"Timestamp {chunk.get('timestamp') or '0:00'}"
            elif chunk.get("source_type") in {"text", "markdown"}:
                location = f"Document section {chunk['page']}"
            else:
                location = f"पृष्ठ/Page {chunk['page']}"
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
            "timestamp_url": c.get("timestamp_url"),
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
            "transcript_coverage_ratio": c.get("transcript_coverage_ratio"),
            "transcript_repeated_snippet_ratio": c.get(
                "transcript_repeated_snippet_ratio"
            ),
            "transcript_devanagari_letter_ratio": c.get(
                "transcript_devanagari_letter_ratio"
            ),
            "transcript_latin_letter_ratio": c.get(
                "transcript_latin_letter_ratio"
            ),
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
