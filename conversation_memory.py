"""Bounded chat context and conservative follow-up detection."""

import re

MAX_HISTORY_EXCHANGES = 12
MAX_HISTORY_CHARACTERS = 24_000
MAX_MESSAGE_CHARACTERS = 6_000


# This is intentionally a conservative gate, not a complete language parser.
# Its job is only to avoid paying for query rewriting when the latest question
# is clearly self-contained. A triggered question is still checked by the
# constrained contextualizer before retrieval.
_CONTEXT_DEPENDENT = (
    # English references and conversational continuations.
    r"\b(?:it|its|they|them|their|he|him|his|she|her|hers|this|that|these|those|"
    r"former|latter|same|above)\b",
    r"^(?:and|also|then|so|but|what about|how about)\b",
    r"^(?:tell me more|explain more|continue|go on|elaborate)(?:\b|$)",
    r"^(?:explain|answer|translate|say|write)(?:\s+(?:it|this|that))?\s+in\s+"
    r"(?:english|hindi|hind[iy]a?|देवनागरी)(?:\b|$)",
    # Devanagari Hindi references and short continuation phrases.
    r"(?:^|\s)(?:यह|ये|वह|वे|इसका|इसके|इसकी|इसे|उसका|उसके|उसकी|उसे|"
    r"उनका|उनके|उनकी|फिर|आगे)(?:\s|$)",
    r"^(?:और बताओ|आगे बताओ|फिर क्या|क्यों|कैसे)(?:\s|$)",
    # Common Romanized-Hindi equivalents.
    r"\b(?:yeh|ye|woh|voh|iska|iske|iski|usse|uska|uske|uski|unka|unke|unki|"
    r"phir|aage)\b",
    r"^(?:aur batao|aage batao|phir kya|kyun|kaise)(?:\s|$)",
)


def bounded_history(messages: list[dict] | None) -> list[dict]:
    """Keep recent text messages only; never mutate the caller's stored history."""
    result = []
    remaining = MAX_HISTORY_CHARACTERS
    for message in reversed((messages or [])[-MAX_HISTORY_EXCHANGES * 2:]):
        if message.get("role") not in {"user", "model"}:
            continue
        text = "\n".join(
            part["text"] for part in message.get("parts", [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text:
            continue
        if remaining <= 0:
            break
        limit = min(MAX_MESSAGE_CHARACTERS, remaining)
        if len(text) > limit:
            marker = "\n[message shortened]"
            if limit <= len(marker):
                break
            text = text[:limit - len(marker)] + marker
        result.append({"role": message["role"], "parts": [{"text": text}]})
        remaining -= len(text)
    result.reverse()
    # Do not begin a shortened model conversation with an orphaned answer.
    if result and result[0]["role"] == "model":
        result.pop(0)
    return result


def needs_contextualization(question: str, history: list[dict]) -> bool:
    """Whether retrieval may need recent turns to resolve the latest question."""
    if not history:
        return False
    normalized = " ".join(question.casefold().split()).strip()
    return any(re.search(pattern, normalized) for pattern in _CONTEXT_DEPENDENT)
