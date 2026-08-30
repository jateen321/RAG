"""Match machine-readable answer citations to validated source metadata."""

from __future__ import annotations

import re


_INLINE_CITATION = re.compile(
    r"(?:⟦([^⟦⟧\n]+)⟧|\[\[([^\[\]\n]+)\]\]|\[([^\[\]\n]+)\])"
)
_CITATION_CONTENT = re.compile(
    r"^(.+),\s*(?:(?:(?:Page|पृष्ठ|पेज)\s+(\d+)"
    r"(?:\s*/\s*(?:Page|पृष्ठ|पेज)\s+\2)?|Document section\s+(\d+))"
    r"|Timestamp\s+(\d{1,2}:\d{2})|(Web))$",
    re.IGNORECASE,
)


def _normalized_name(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _source_key(source: dict) -> tuple:
    chunk_id = source.get("chunk_id")
    if chunk_id:
        return ("chunk", chunk_id)
    return (
        "location",
        _normalized_name(source.get("source")),
        source.get("page"),
        source.get("timestamp"),
        source.get("source_url"),
    )


def _matches(content: str, source: dict) -> bool:
    match = _CITATION_CONTENT.fullmatch(content.strip())
    if not match:
        return False
    cited_name = _normalized_name(match.group(1))
    source_names = {
        _normalized_name(source.get(field))
        for field in ("source", "video_title", "citation_label")
        if source.get(field)
    }
    if cited_name not in source_names:
        return False

    cited_page = match.group(2) or match.group(3)
    cited_timestamp = match.group(4)
    cited_web = bool(match.group(5))
    if cited_web:
        return source.get("source_type") == "web"
    if cited_timestamp:
        return source.get("timestamp") == cited_timestamp
    try:
        return int(cited_page) == int(source.get("page"))
    except (TypeError, ValueError):
        return False


def align_answer_sources(
    answer: str,
    current_sources: list[dict] | None,
    historical_sources: list[dict] | None = None,
) -> list[dict]:
    """Return validated sources in citation order.

    Current retrieval results take precedence when both current and historical
    metadata describe the same citation. If no citation can be matched, retain
    the current results so an uncited answer still has supporting evidence.
    """
    current = list(current_sources or [])
    candidates = [*current, *(historical_sources or [])]
    aligned = []
    seen = set()
    for token in _INLINE_CITATION.finditer(answer or ""):
        content = next(group for group in token.groups() if group is not None)
        for source in candidates:
            if not _matches(content, source):
                continue
            key = _source_key(source)
            if key not in seen:
                aligned.append(source)
                seen.add(key)
            break
    return aligned or current
