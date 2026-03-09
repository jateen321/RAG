"""
📚 Hindi Textbook RAG Application
==================================
Query your scanned Hindi textbook PDFs using AI.
Uses free Google Gemini API for embeddings and answers.

Usage:
    python app.py index <pdf_file>       Index a PDF for searching
    python app.py ask "your question"    Ask a one-shot question
    python app.py chat                   Start interactive chat
    python app.py status                 Show database statistics
    python app.py reset                  Clear all indexed data
"""

import sys
import os
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()


def print_banner():
    """Print a nice welcome banner."""
    banner = """
[bold cyan]📚 Hindi Textbook RAG[/bold cyan]
[dim]Query your scanned Hindi textbooks using AI[/dim]
[dim]Powered by Google Gemini (Free) + EasyOCR[/dim]
    """
    console.print(Panel(banner.strip(), border_style="cyan"))


def cmd_index(pdf_path: str):
    """Index a PDF file."""
    if not os.path.exists(pdf_path):
        console.print(f"[red]❌ File not found: {pdf_path}[/red]")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        console.print("[red]❌ Please provide a PDF file.[/red]")
        sys.exit(1)

    from ocr_engine import extract_text_from_pdf
    from indexer import index_document

    # Step 1: OCR
    console.print("\n[bold]Step 1/2: Extracting text (OCR)...[/bold]")
    pages_text = extract_text_from_pdf(pdf_path)

    if not pages_text:
        console.print("[red]❌ No text could be extracted from this PDF.[/red]")
        console.print("   The PDF might be empty or the scan quality is too low.")
        sys.exit(1)

    # Step 2: Index
    console.print("\n[bold]Step 2/2: Indexing for search...[/bold]")
    source_name = os.path.basename(pdf_path)
    num_chunks = index_document(pages_text, source_name)

    if num_chunks > 0:
        console.print(Panel(
            f"[green]🎉 Successfully indexed![/green]\n\n"
            f"File: [bold]{source_name}[/bold]\n"
            f"Pages processed: {len(pages_text)}\n"
            f"Chunks created: {num_chunks}\n\n"
            f"[dim]Now try:[/dim]\n"
            f"  [bold]python app.py ask \"इस किताब में क्या है?\"[/bold]\n"
            f"  [bold]python app.py chat[/bold]",
            title="✅ Done",
            border_style="green",
        ))


def cmd_ask(question: str):
    """Ask a single question."""
    from rag_engine import ask

    console.print(f"\n🔍 [bold]Question:[/bold] {question}\n")
    answer = ask(question)

    console.print(Panel(
        Markdown(answer),
        title="🤖 Answer",
        border_style="green",
        padding=(1, 2),
    ))


def cmd_chat():
    """Start interactive chat session."""
    from rag_engine import ask

    print_banner()

    from indexer import get_stats
    stats = get_stats()

    if stats["total_chunks"] == 0:
        console.print("[red]❌ No documents indexed yet![/red]")
        console.print("   Run: [bold]python app.py index <pdf_file>[/bold] first.\n")
        return

    console.print(f"[green]📊 {stats['total_chunks']} chunks in database[/green]")
    console.print("[dim]Type your questions in Hindi or English. Type 'quit' to exit.[/dim]\n")

    chat_history = []

    while True:
        try:
            question = Prompt.ask("[bold cyan]📝 You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]👋 Goodbye![/dim]")
            break

        question = question.strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q", "बाहर"):
            console.print("[dim]👋 Goodbye! Happy learning! 📖[/dim]")
            break

        # Get answer
        answer = ask(question, chat_history=chat_history)

        console.print(Panel(
            Markdown(answer),
            title="🤖 Answer",
            border_style="green",
            padding=(1, 2),
        ))

        # Maintain chat history (keep last 6 messages for context)
        chat_history.append({"role": "user", "parts": [question]})
        chat_history.append({"role": "model", "parts": [answer]})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

        console.print()


def cmd_status():
    """Show database statistics."""
    from indexer import get_stats

    stats = get_stats()
    console.print(Panel(
        f"📊 [bold]Database Statistics[/bold]\n\n"
        f"Total chunks indexed: [bold]{stats['total_chunks']}[/bold]\n"
        f"Database path: [dim]{stats['db_path']}[/dim]",
        border_style="cyan",
    ))


def cmd_reset():
    """Clear all indexed data."""
    from config import CHROMA_DB_PATH

    if os.path.exists(CHROMA_DB_PATH):
        confirm = Prompt.ask(
            "[yellow]⚠️  Delete all indexed data?[/yellow]",
            choices=["yes", "no"],
            default="no",
        )
        if confirm == "yes":
            shutil.rmtree(CHROMA_DB_PATH)
            console.print("[green]✅ All indexed data cleared.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")
    else:
        console.print("[dim]No indexed data found.[/dim]")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_banner()
        console.print("""
[bold]Usage:[/bold]

  [cyan]python app.py index <pdf_file>[/cyan]    Index a PDF for searching
  [cyan]python app.py ask "question"[/cyan]      Ask a one-shot question  
  [cyan]python app.py chat[/cyan]                Start interactive chat
  [cyan]python app.py status[/cyan]              Show database statistics
  [cyan]python app.py reset[/cyan]               Clear all indexed data

[bold]Examples:[/bold]

  python app.py index data/history_book.pdf
  python app.py ask "मुगल साम्राज्य कब शुरू हुआ?"
  python app.py chat
""")
        return

    command = sys.argv[1].lower()

    if command == "index":
        if len(sys.argv) < 3:
            console.print("[red]❌ Please provide a PDF file path.[/red]")
            console.print("   Usage: python app.py index <pdf_file>")
            sys.exit(1)
        cmd_index(sys.argv[2])

    elif command == "ask":
        if len(sys.argv) < 3:
            console.print("[red]❌ Please provide a question.[/red]")
            console.print('   Usage: python app.py ask "your question here"')
            sys.exit(1)
        cmd_ask(" ".join(sys.argv[2:]))

    elif command == "chat":
        cmd_chat()

    elif command == "status":
        cmd_status()

    elif command == "reset":
        cmd_reset()

    else:
        console.print(f"[red]❌ Unknown command: {command}[/red]")
        console.print("   Valid commands: index, ask, chat, status, reset")
        sys.exit(1)


if __name__ == "__main__":
    main()
