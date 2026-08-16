"""
📚 Hindi Textbook RAG Application
==================================
Query your scanned Hindi textbook PDFs using AI.
Uses free Google Gemini API for embeddings and answers.

Usage:
    python app.py index                  Pick a PDF from data/ and index it
    python app.py index <pdf_file>       Index a specific PDF
    python app.py ask "your question"    Ask a one-shot question
    python app.py  chat                   Start interactive chat
    python app.py status                 Show database statistics
    python app.py remove <source>        Delete one document from the index
    python app.py reset                  Clear all indexed data
"""

import sys
import os
import shutil

# Importing readline is the whole fix: Python wires it into input() on import,
# which gives every prompt arrow-key editing and ↑/↓ history. Without it, arrow
# keys arrive as raw escape codes (^[[D) and end up inside the typed text.
try:
    import readline  # noqa: F401  — imported for its side effect only
except ImportError:
    pass  # Windows has no readline; prompts still work, just without editing

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def print_banner():
    """Print a nice welcome banner."""
    banner = """
[bold cyan]📚 Hindi Textbook RAG[/bold cyan]
[dim]Query your scanned Hindi textbooks using AI[/dim]
[dim]Powered by Google Gemini (Free) + EasyOCR[/dim]
    """
    console.print(Panel(banner.strip(), border_style="cyan"))


def _ask(*args, **kwargs) -> str:
    """Prompt.ask, but Ctrl+C / Ctrl+D exits cleanly instead of dumping a traceback."""
    try:
        return Prompt.ask(*args, **kwargs)
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(1)


def _list_data_pdfs() -> list:
    """Every PDF sitting in the data/ folder, sorted by name."""
    from config import DATA_DIR

    if not os.path.isdir(DATA_DIR):
        return []
    names = sorted(n for n in os.listdir(DATA_DIR) if n.lower().endswith(".pdf"))
    return [os.path.join(DATA_DIR, n) for n in names]


def _page_count(pdf_path: str) -> str:
    """Page count for the picker table ('—' if the file can't be opened)."""
    try:
        import pymupdf

        doc = pymupdf.open(pdf_path)
        count = len(doc)
        doc.close()
        return f"{count} pages"
    except Exception:
        return "—"


def _pick_pdf() -> str:
    """Show a numbered list of data/*.pdf and return the chosen path.

    Filenames like "027. Bhagya Likhne Ki Kalam - Karm.pdf" are painful to type
    exactly, so picking by number avoids the whole class of typo errors.
    """
    pdfs = _list_data_pdfs()
    if not pdfs:
        console.print("[red]❌ No PDFs found in the data/ folder.[/red]")
        console.print("   Drop a PDF into [cyan]data/[/cyan] and try again.")
        sys.exit(1)

    table = Table(title="📚 PDFs in data/", border_style="cyan", header_style="bold")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Document", style="green", overflow="fold")
    table.add_column("Pages", justify="right", style="dim")

    for i, path in enumerate(pdfs, 1):
        table.add_row(str(i), os.path.basename(path), _page_count(path))

    console.print()
    console.print(table)
    console.print()

    choice = _ask(
        f"Select a PDF [cyan][1-{len(pdfs)}][/cyan]",
        choices=[str(i) for i in range(1, len(pdfs) + 1)],
        show_choices=False,
    )
    return pdfs[int(choice) - 1]


def _resolve_or_suggest(pdf_path: str) -> str:
    """A path that doesn't exist — try to recover instead of just failing.

    Handles the common cases: right name but wrong case, or a near-miss typo
    (e.g. "...karma.pdf" for "...karm.pdf"). Falls back to the picker.
    """
    import difflib

    pdfs = _list_data_pdfs()
    names = [os.path.basename(p) for p in pdfs]
    typed = os.path.basename(pdf_path)

    console.print(f"[red]❌ File not found: {pdf_path}[/red]")

    close = difflib.get_close_matches(typed, names, n=3, cutoff=0.5)
    if close:
        console.print("\n[yellow]Did you mean:[/yellow]")
        for name in close:
            console.print(f"   [green]{name}[/green]")

    if not pdfs:
        sys.exit(1)

    console.print()
    if _ask(
        "Pick from data/ instead?", choices=["yes", "no"], default="yes"
    ) != "yes":
        sys.exit(1)
    return _pick_pdf()


def cmd_index(pdf_path: str = None):
    """Index a PDF file. With no path given, show the picker."""
    if pdf_path is None:
        pdf_path = _pick_pdf()
    elif not os.path.exists(pdf_path):
        pdf_path = _resolve_or_suggest(pdf_path)

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
            # Raw Prompt.ask here — chat handles Ctrl+C/Ctrl+D itself, with a
            # friendly goodbye instead of _ask's "Cancelled." + exit(1).
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
        # `parts` must hold Part-shaped dicts, not bare strings — same shape the
        # live message uses in rag_engine.ask (a plain str fails SDK validation).
        chat_history.append({"role": "user", "parts": [{"text": question}]})
        chat_history.append({"role": "model", "parts": [{"text": answer}]})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

        console.print()


def cmd_status():
    """Show database statistics."""
    from indexer import get_stats

    stats = get_stats()
    documents = stats.get("documents", [])

    console.print(Panel(
        f"📊 [bold]Database Statistics[/bold]\n\n"
        f"Documents indexed: [bold]{len(documents)}[/bold]\n"
        f"Total chunks indexed: [bold]{stats['total_chunks']}[/bold]\n"
        f"Database path: [dim]{stats['db_path']}[/dim]",
        border_style="cyan",
    ))

    if not documents:
        console.print(
            "\n[yellow]No documents indexed yet.[/yellow] "
            "Run [cyan]python app.py index <pdf>[/cyan] to add one.\n"
        )
        return

    table = Table(title="📚 Indexed Documents", border_style="cyan", header_style="bold")
    table.add_column("Document", style="green", overflow="fold")
    table.add_column("Pages", justify="right")
    table.add_column("Page range", justify="center", style="dim")
    table.add_column("Chunks", justify="right")

    for doc in documents:
        first, last = doc["first_page"], doc["last_page"]
        page_range = f"{first}–{last}" if first is not None else "—"
        table.add_row(
            doc["source"],
            str(doc["pages"]),
            page_range,
            str(doc["chunks"]),
        )

    console.print()
    console.print(table)
    console.print()


def cmd_remove(source_name: str):
    """Delete one document's chunks, leaving the rest of the index intact."""
    from indexer import get_stats, remove_document

    indexed = {d["source"]: d for d in get_stats().get("documents", [])}

    if source_name not in indexed:
        console.print(f"[red]❌ '{source_name}' is not indexed.[/red]")
        if indexed:
            console.print("\n[bold]Indexed documents:[/bold]")
            for name in indexed:
                console.print(f"   [green]{name}[/green]")
            console.print("\n[dim]Use the exact name shown above.[/dim]")
        else:
            console.print("[dim]Nothing is indexed yet.[/dim]")
        sys.exit(1)

    doc = indexed[source_name]
    confirm = _ask(
        f"[yellow]⚠️  Delete [bold]{doc['chunks']}[/bold] chunks "
        f"({doc['pages']} pages) from '{source_name}'?[/yellow]",
        choices=["yes", "no"],
        default="no",
    )
    if confirm != "yes":
        console.print("[dim]Cancelled.[/dim]")
        return

    deleted = remove_document(source_name)
    console.print(f"[green]✅ Removed {deleted} chunks from '{source_name}'.[/green]")
    console.print("[dim]Run `python app.py status` to see the updated index.[/dim]")


def cmd_reset():
    """Clear all indexed data."""
    from config import CHROMA_DB_PATH

    if os.path.exists(CHROMA_DB_PATH):
        confirm = _ask(
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

  [cyan]python app.py index[/cyan]               Pick a PDF from data/ and index it
  [cyan]python app.py index <pdf_file>[/cyan]    Index a specific PDF
  [cyan]python app.py ask "question"[/cyan]      Ask a one-shot question  
  [cyan]python app.py chat[/cyan]                Start interactive chat
  [cyan]python app.py status[/cyan]              Show database statistics
  [cyan]python app.py remove <source>[/cyan]     Delete one document from the index
  [cyan]python app.py reset[/cyan]               Clear all indexed data

[bold]Examples:[/bold]

  python app.py index data/CIL.pdf
  python app.py ask "Where is Coal India Limited's corporate headquarters?"
  python app.py remove CIL.pdf
  python app.py chat
""")
        return

    command = sys.argv[1].lower()

    if command == "index":
        # No path given → show the picker instead of erroring out.
        cmd_index(sys.argv[2] if len(sys.argv) > 2 else None)

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

    elif command == "remove":
        if len(sys.argv) < 3:
            console.print("[red]❌ Please provide the document name to remove.[/red]")
            console.print("   Usage: python app.py remove <source>")
            console.print("   [dim]Tip: run `python app.py status` to see indexed names.[/dim]")
            sys.exit(1)
        cmd_remove(sys.argv[2])

    elif command == "reset":
        cmd_reset()

    else:
        console.print(f"[red]❌ Unknown command: {command}[/red]")
        console.print("   Valid commands: index, ask, chat, status, remove, reset")
        sys.exit(1)


if __name__ == "__main__":
    main()
