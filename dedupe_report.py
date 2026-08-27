"""Report (and optionally remove) documents indexed more than once.

Three ingestion paths spelled the same file differently -- "CIL.pdf" from the
CLI, "data/CIL.pdf" from index_folder, a data-root-relative path from the API --
and until the basename guard landed in indexer.is_document_indexed, nothing
stopped a second copy being written. Duplicate chunks compete for top-k slots
and understate source_precision (measured: 0.8105 -> 0.8421 on the v2 set).

DRY RUN BY DEFAULT. Deleting is destructive, the collection represents real
embedding spend, and a concurrent indexing run may be writing to it -- so
--apply is opt-in and prints what it will do before doing it.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from rich.console import Console

console = Console()


def _groups() -> dict[str, list[dict]]:
    """Basename -> one entry per distinct (source_name, document_id) copy."""
    from indexer import _get_collection

    metadatas = _get_collection().get(include=["metadatas"])["metadatas"] or []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for md in metadatas:
        if not md:
            continue
        counts[(md.get("source_name", ""), md.get("document_id", ""))] += 1

    by_base: dict[str, list[dict]] = defaultdict(list)
    for (source_name, document_id), chunks in counts.items():
        by_base[os.path.basename(source_name).casefold()].append(
            {"source_name": source_name, "document_id": document_id, "chunks": chunks}
        )
    return {b: v for b, v in by_base.items() if len(v) > 1}


def _keep(copies: list[dict]) -> dict:
    """Keep the most complete copy; break ties on the longer stored name.

    Chunk count is the completeness signal: a run killed by quota leaves a
    partial document, and the Gita's 857-vs-240 split is exactly that. The
    longer name is preferred on a tie because it carries folder context.
    """
    return max(copies, key=lambda c: (c["chunks"], len(c["source_name"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete the redundant copies (default: report only).",
    )
    args = parser.parse_args()

    groups = _groups()
    if not groups:
        console.print("[green]✅ No duplicated documents found.[/green]")
        return 0

    total_removable = 0
    console.print(f"[bold]{len(groups)} document(s) indexed more than once[/bold]\n")
    for base, copies in sorted(groups.items()):
        keeper = _keep(copies)
        console.print(f"  [bold]{base}[/bold]")
        for c in sorted(copies, key=lambda c: -c["chunks"]):
            mark = "[green]KEEP  [/green]" if c is keeper else "[red]REMOVE[/red]"
            if c is not keeper:
                total_removable += c["chunks"]
            console.print(
                f"    {mark} {c['chunks']:5} chunks  doc_id={c['document_id']}  "
                f"{c['source_name']!r}"
            )
        console.print("")

    console.print(f"[bold]{total_removable} chunk(s) would be removed.[/bold]")

    if not args.apply:
        console.print("\n[yellow]Dry run — nothing was deleted.[/yellow]")
        console.print("   Re-run with [bold]--apply[/bold] to remove the copies marked REMOVE.")
        return 0

    from indexer import _get_collection

    collection = _get_collection()
    for base, copies in sorted(groups.items()):
        keeper = _keep(copies)
        for c in copies:
            if c is keeper:
                continue
            collection.delete(where={"document_id": c["document_id"]})
            console.print(f"   [red]removed[/red] {c['chunks']:5} chunks — {c['source_name']!r}")
    console.print(f"\n[green]✅ Removed {total_removable} chunk(s).[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
