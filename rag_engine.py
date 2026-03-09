"""
RAG Engine — Orchestrates retrieval and generation.

Retrieves relevant chunks, builds a context-aware prompt,
and generates answers using Gemini Flash (free).
"""

from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from config import GEMINI_API_KEY, LLM_MODEL
from retriever import retrieve

console = Console()

# Initialize Gemini client (new SDK)
_client = genai.Client(api_key=GEMINI_API_KEY)

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
            preview = chunk["text"][:80].replace("\n", " ") + "..."
            console.print(f"   [dim]• पृष्ठ/Page {page}: {preview}[/dim]")
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
