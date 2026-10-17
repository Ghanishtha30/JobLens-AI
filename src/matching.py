"""Skill matching and the TF-IDF text-similarity signal.

Two separate things live here:
  1. Skill-based matching  - exact set comparison after normalisation (the main result).
  2. Text similarity       - TF-IDF + cosine similarity, shown as a supplementary signal.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # scikit-learn is a required dependency, but fail softly so the app can still run
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

from .utils import Taxonomy, normalize_key

_GENERIC_SUFFIXES = (
    "programming language", "programming", "language", "skills", "skill",
    "development", "basics", "fundamentals",
)
_MAX_SKILL_LENGTH = 60
_SUGGESTION_CUTOFF = 0.65  # suggestions only ('Did you mean...?'); never applied automatically


@dataclass
class NormalizedProfile:
    """The user's skills after normalisation."""

    taxonomy_skills: list = field(default_factory=list)   # recognised, canonical names
    custom_skills: list = field(default_factory=list)     # not in the taxonomy, kept as typed
    duplicates_removed: int = 0
    suggestions: dict = field(default_factory=dict)       # unknown input -> closest taxonomy skill
    ignored: list = field(default_factory=list)           # empty or over-long entries

    @property
    def all_skills(self) -> list:
        return self.taxonomy_skills + self.custom_skills


@dataclass
class MatchResult:
    matched: list
    missing: list
    additional: list


def resolve_skill(raw: str, taxonomy: Taxonomy) -> Optional[str]:
    """Map free text ('python', 'Python programming', 'PYTHON') to a canonical skill name."""
    key = normalize_key(raw)
    if not key:
        return None
    if key in taxonomy.alias_lookup:
        return taxonomy.alias_lookup[key]
    current = key
    changed = True
    while changed:
        changed = False
        for suffix in _GENERIC_SUFFIXES:
            if current.endswith(" " + suffix):
                current = current[: -(len(suffix) + 1)].strip()
                if current in taxonomy.alias_lookup:
                    return taxonomy.alias_lookup[current]
                changed = True
                break
    return None


def normalize_user_skills(raw_items, taxonomy: Taxonomy) -> NormalizedProfile:
    """Normalise a list of user-entered skills (items may themselves be comma separated)."""
    profile = NormalizedProfile()
    seen: set = set()
    for item in raw_items or []:
        for piece in re.split(r"[,;\n|]+", str(item)):
            piece = " ".join(piece.split())
            if not piece:
                continue
            if len(piece) > _MAX_SKILL_LENGTH:
                profile.ignored.append(piece[:30] + "...")
                continue
            canonical = resolve_skill(piece, taxonomy)
            if canonical is not None:
                if canonical.lower() in seen:
                    profile.duplicates_removed += 1
                    continue
                seen.add(canonical.lower())
                profile.taxonomy_skills.append(canonical)
            else:
                key = normalize_key(piece)
                if not key:
                    profile.ignored.append(piece)
                    continue
                if key in seen:
                    profile.duplicates_removed += 1
                    continue
                seen.add(key)
                profile.custom_skills.append(piece)
                close = []
                if len(key) >= 4:  # very short strings give meaningless fuzzy matches
                    close = difflib.get_close_matches(key, list(taxonomy.alias_lookup), n=1, cutoff=_SUGGESTION_CUTOFF)
                close = [c for c in close if c[0] == key[0]]  # typos rarely change the first letter
                if close:
                    profile.suggestions[piece] = taxonomy.alias_lookup[close[0]]
    return profile


def match_skills(job_skills: list, user_skills: list) -> MatchResult:
    """Split skills into matched / missing (job order kept) and additional (user order kept)."""
    user_keys = {s.lower() for s in user_skills}
    job_keys = {s.lower() for s in job_skills}
    return MatchResult(
        matched=[s for s in job_skills if s.lower() in user_keys],
        missing=[s for s in job_skills if s.lower() not in user_keys],
        additional=[s for s in user_skills if s.lower() not in job_keys],
    )


def _expand(skill: str, taxonomy: Taxonomy) -> str:
    """Text used to represent a skill: name, aliases and its category as ONE token.

    The category token (e.g. 'data_science_ml') is what lets related skills such as Pandas and
    NumPy share vocabulary, which is the extra information TF-IDF adds beyond exact matching.
    """
    entry = taxonomy.skills.get(skill)
    if entry is None:
        return skill
    category_token = re.sub(r"[^a-z0-9]+", "_", entry.category.lower()).strip("_")
    return " ".join([entry.name, *entry.aliases, category_token])


def tfidf_similarity(job_skills: list, user_skills: list, taxonomy: Taxonomy) -> Optional[float]:
    """Cosine similarity (0-1) between the job's skill text and the user's skill text.

    The TF-IDF vocabulary is learned from the whole taxonomy plus the two documents, so rare
    terms weigh more than common category words. Returns None if scikit-learn is unavailable
    or the vocabulary is empty, and 0.0 if either side has no skills.
    """
    if TfidfVectorizer is None:
        return None
    if not job_skills or not user_skills:
        return 0.0
    corpus = [_expand(name, taxonomy) for name in taxonomy.names()]
    job_doc = " ".join(_expand(s, taxonomy) for s in job_skills)
    user_doc = " ".join(_expand(s, taxonomy) for s in user_skills)
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z][A-Za-z0-9+#._]*")
    try:
        vectorizer.fit(corpus + [job_doc, user_doc])
        matrix = vectorizer.transform([job_doc, user_doc])
    except ValueError:
        return None
    similarity = cosine_similarity(matrix[0], matrix[1])[0, 0]
    return float(np.clip(similarity, 0.0, 1.0))
