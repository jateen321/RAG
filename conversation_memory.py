"""Bounded chat context and conservative routing for explicit recall requests."""

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


# Full matches are intentional: "What did Arjuna ask earlier?" and mixed
# requests about books must not be routed away from document retrieval.
_PREVIOUS_EN = (
    r"what (?:was|is) (?:my|the) (?:last|previous|earlier) (?:query|question|message)(?: of (?:mine|my))?",
    r"what did i (?:ask|say)(?: you)? (?:earlier|before|last|previously)",
    r"(?:repeat|show|tell)(?: me)? my (?:last|previous|earlier) (?:query|question|message)",
    r"what was the (?:last|previous) (?:question|query) i asked(?: you)?",
)
_PREVIOUS_HI = (
    r"(?:मेरा|मेरी) (?:पिछला|पिछली|पहले का) (?:सवाल|प्रश्न|क्वेरी) क्या (?:था|थी|है)",
    r"मैंने (?:पहले|पिछली बार|अभी) क्या पूछा(?: था)?",
    r"(?:mera|meri) (?:pichla|pichhla|pichli|pichhli|pehle ka) (?:sawal|savaal|prashn|question|query) kya (?:tha|thi|hai)",
    r"maine (?:pehle|pichli baar|abhi) kya (?:pucha|poocha)(?: tha)?",
)
_HISTORY_EN = (
    r"(?:summarize|summarise|recap)(?: our| this| my| the)? (?:conversation|chat|discussion)(?: so far)?",
    r"what (?:have we|did we) (?:discuss|discussed|talk about)(?: so far| earlier)?",
    r"(?:list|show|repeat)(?: me)? my (?:previous|earlier|last|recent) (?:questions|queries|messages)",
    r"what (?:were|are) my (?:previous|earlier|last|recent) (?:questions|queries|messages)",
)
_HISTORY_HI = (
    r"(?:हमारी|इस) (?:बातचीत|चैट) का सारांश (?:बताओ|बताएं|दो)",
    r"हमने (?:अब तक|पहले) क्या (?:बात की|चर्चा की)(?: थी|है)?",
    r"मेरे (?:पिछले|पहले के) (?:सवाल|प्रश्न) (?:बताओ|बताएं|क्या थे)",
    r"(?:hamari|humari|is) (?:baatcheet|baat cheet|chat) ka (?:summary|saransh) (?:batao|bataiye|do)",
)


def recall_intent(question: str) -> tuple[str, str] | None:
    normalized = " ".join(question.casefold().split()).strip(" ?!.।")
    for intent, language, patterns in (
        ("previous", "en", _PREVIOUS_EN), ("previous", "hi", _PREVIOUS_HI),
        ("summary", "en", _HISTORY_EN), ("summary", "hi", _HISTORY_HI),
    ):
        if any(re.fullmatch(pattern, normalized) for pattern in patterns):
            return intent, language
    return None


def previous_question(history: list[dict]) -> str | None:
    for message in reversed(history):
        if message["role"] == "user":
            return message["parts"][0]["text"]
    return None
