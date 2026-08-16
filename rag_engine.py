"""
RAG Engine — Orchestrates retrieval and generation.

Retrieves relevant chunks, builds a context-aware prompt,
and generates answers using Gemini Flash (free).
"""

from typing import NoReturn

from google.genai import types
from google.genai.errors import ClientError
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.markup import escape

from config import LLM_MODEL
from llm_client import get_client
from retriever import retrieve

console = Console()

# Initialize Gemini client (backend chosen in config: Developer API or Vertex)
_client = get_client()

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful study assistant for Hindi textbooks. Your job is to help students learn and understand the content from their books.

RULES:
1. Answer ONLY based on the provided context from the textbook.
2. If the context doesn't contain enough information, say so honestly.
3. Reply in the SAME LANGUAGE as the user's question (Hindi or English).
4. Always mention the page number(s) where you found the information.
5. Explain concepts clearly, as if teaching a student.
6. If asked to summarize, provide a clear and concise summary.
7. Use bullet points and formatting to make answers easy to read."""


def ask(question: str, chat_history: list = None, show_sources: bool = True) -> str:
    """
    Answer a question using RAG (Retrieve + Generate).

    Args:
        question: User's question in Hindi or English.
        chat_history: Optional list of previous messages for context.
        show_sources: Whether to display source chunks.

    Returns:
        The generated answer string.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question)

    if not chunks:
        return "❌ कोई प्रासंगिक जानकारी नहीं मिली। कृपया पहले एक PDF इंडेक्स करें।\n(No relevant information found. Please index a PDF first.)"

    # Step 2: Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[पृष्ठ {chunk['page']} / Page {chunk['page']}]:\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Show sources if requested
    if show_sources:
        console.print("\n[dim]📚 Sources found:[/dim]")
        for chunk in chunks:
            page = chunk["page"]
            source = chunk["source"]
            # escape(): the preview is raw OCR text, and console.print reads
            # `[...]` as style markup — an unescaped bracket would be eaten
            # (or raise MarkupError) instead of being shown.
            preview = escape(chunk["text"][:80].replace("\n", " ")) + "..."
            console.print(
                f"   • [cyan]{escape(source)}[/cyan] "
                f"[dim]पृष्ठ/Page {page}:[/dim] [dim]{preview}[/dim]"
            )
        console.print()

    # Step 3: Build the prompt
    user_message = f"""Context from the textbook:

{context}

---

Student's question: {question}

Please answer based on the context above."""

    # Build message list for multi-turn chat
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "parts": [{"text": user_message}]})

    # Step 4: Generate answer with Gemini (new SDK)
    try:
        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        answer = response.text
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            answer = "⏳ Rate limit reached. Please wait a minute and try again."
        else:
            answer = f"❌ Error generating answer: {str(e)}"

    return answer


def ask_simple(question: str) -> str:
    """Simple one-shot question without chat history or source display."""
    return ask(question, chat_history=None, show_sources=False)


def _quota_guard(exc: ClientError) -> NoReturn:
    """Translate a Gemini 429 (quota/rate limit) into a RuntimeError, which the
    API layer maps to a 503. Any other ClientError propagates unchanged."""
    if exc.code == 429:
        raise RuntimeError(
            "Gemini quota/rate limit reached. Please try again later."
        ) from exc
    raise exc


def ask_with_sources(question: str) -> dict:
    """
    Answer a question and return the answer together with its sources.

    Programmatic sibling of ``ask`` (no rich console output). Used by the
    FastAPI ``/ask`` route and the evaluation harness.

    Returns:
        {"answer": str, "sources": [{"page", "source", "distance", "preview"}]}

    Raises:
        RuntimeError: if nothing is indexed yet (empty / missing collection),
            so the API can surface a 503.
    """
    try:
        chunks = retrieve(question)      # embedding call — can raise a 429
    except ClientError as e:
        _quota_guard(e)                  # 429 → RuntimeError; else re-raised

    if not chunks:
        raise RuntimeError(
            "No indexed documents found. Index a PDF before asking questions."
        )

    # Build the grounding context (same layout as `ask`).
    context = "\n\n---\n\n".join(
        f"[पृष्ठ {c['page']} / Page {c['page']}]:\n{c['text']}" for c in chunks
    )
    user_message = f"""Context from the textbook:

{context}

---

Student's question: {question}

Please answer based on the context above."""

    try:
        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=[{"role": "user", "parts": [{"text": user_message}]}],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        answer = response.text
    except ClientError as e:
        _quota_guard(e)                  # 429 → RuntimeError; else re-raised

    sources = [
        {
            "page": c["page"],
            "source": c["source"],
            "distance": c["distance"],
            "preview": c["text"][:120].replace("\n", " "),
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}
