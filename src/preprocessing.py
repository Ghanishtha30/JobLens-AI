"""Text preprocessing for job descriptions.

Design rule: never strip characters that are part of skill names ('+', '#', '.', '/', '-'),
so 'C++', 'C#', 'Node.js' and 'CI/CD' survive. Original casing is kept for segments because
a few ambiguous skills (R, C, Excel) need case-sensitive matching; lowercasing is only used
for tokenization and keyword statistics.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

MIN_CHARS = 40
MIN_WORDS = 10
MAX_CHARS = 20_000


class InputError(ValueError):
    """Invalid user input. The message is safe to show directly in the UI."""


# Common English words plus generic job-ad filler; used only for the keyword statistics.
STOPWORDS = frozenset(
    """a about above after all also am an and any are as at be been being both but by can could
    did do does doing for from had has have having he her his how i if in into is it its just may
    me more most must my no nor not of off on once only or other our out over own per same she
    should so some such than that the their them then there these they this those through to too
    under until up us very was we were what when where which while who will with within would you
    your role job work working team company looking ability strong good experience skills skill
    knowledge years year candidate candidates responsibilities requirements required preferred
    including etc well new using use used join based related understanding hands-on""".split()
)

_BULLET_RE = re.compile(r"^\s*(?:[-\u2022*\u00b7\u25aa\u25cf\u25e6\u25ba\u2013\u2014]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEADING_RE = re.compile(r"^\s*([^:\n]{2,60}):\s*(.*)$")
_REQUIRED_HEADING = re.compile(
    r"\b(requirements?|qualifications?|must[\s-]have|required|eligibility|"
    r"what you(?:'|\u2019)?ll need|what we(?:'|\u2019)?re looking for)\b", re.IGNORECASE)
_PREFERRED_HEADING = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|good[\s-]to[\s-]have|bonus|desirable|optional|plus)\b",
    re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.]*(?:[-/][a-z0-9+#.]+)*")


@dataclass(frozen=True)
class Segment:
    """One sentence/bullet of the job description plus the section it sits under."""

    text: str
    section: str  # "required", "preferred" or "general"


def validate_job_description(text) -> str:
    """Return the stripped job description or raise InputError with a helpful message."""
    if text is None or not str(text).strip():
        raise InputError("Please paste a job description before clicking Analyze.")
    text = str(text).strip()
    words = len(text.split())
    if len(text) < MIN_CHARS or words < MIN_WORDS:
        raise InputError(
            f"The job description is too short ({words} words). Paste the full text "
            f"(at least {MIN_WORDS} words) so skills can be detected reliably."
        )
    if len(text) > MAX_CHARS:
        raise InputError(
            f"The job description is too long ({len(text):,} characters). "
            f"Please keep it under {MAX_CHARS:,} characters."
        )
    return text


def normalize_whitespace(text: str) -> str:
    """Unify line endings, quotes and spaces; keep line breaks (they carry structure)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def clean_text(text: str) -> str:
    """Lowercased, whitespace-normalized text with bullet characters removed."""
    text = normalize_whitespace(text)
    lines = [_BULLET_RE.sub("", line) for line in text.split("\n")]
    return re.sub(r"\s+", " ", " ".join(lines)).lower().strip()


def tokenize(text: str) -> list:
    """Split text into lowercase tokens, keeping 'c++', 'c#', 'node.js', 'ci/cd'."""
    return [tok.rstrip(".") for tok in _TOKEN_RE.findall(text.lower()) if tok.rstrip(".")]


def _section_from_heading(heading: str) -> str:
    if _PREFERRED_HEADING.search(heading):
        return "preferred"
    if _REQUIRED_HEADING.search(heading):
        return "required"
    return "general"


def split_into_segments(text: str) -> list:
    """Split a job description into sentence-level Segments tagged with their section.

    Headings ('Requirements:', 'Nice to have:', ...) change the section for the lines below
    them; this lets priority rules recognise 'required' and 'preferred' blocks even when the
    individual bullets do not repeat the word.
    """
    segments: list = []
    section = "general"
    for line in normalize_whitespace(text).split("\n"):
        line = _BULLET_RE.sub("", line).strip()
        if not line:
            continue
        match = _HEADING_RE.match(line)
        if match:
            prefix, rest = match.group(1), match.group(2).strip()
            heading_section = _section_from_heading(prefix)
            if len(prefix.split()) <= 8 and (not rest or heading_section != "general"):
                section = heading_section
                line = rest
                if not line:
                    continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if sentence:
                segments.append(Segment(text=sentence, section=section))
    return segments


def top_terms(text: str, exclude: set, limit: int = 10, min_count: int = 2) -> list:
    """Most frequent meaningful words that are NOT known skills.

    Used to show the user terms that may deserve a place in the taxonomy.
    """
    counts: Counter = Counter()
    for token in tokenize(text):
        if len(token) < 3 or token in STOPWORDS or token in exclude or token.isdigit():
            continue
        counts[token] += 1
    return [(term, n) for term, n in counts.most_common(limit * 3) if n >= min_count][:limit]
