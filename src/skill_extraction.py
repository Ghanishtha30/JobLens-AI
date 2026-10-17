"""Dictionary-based skill extraction.

Method (not machine learning): every skill in data/skills.json is turned into a regular
expression built from its name and aliases. The regex uses boundaries so 'Java' does not match
inside 'JavaScript', 'SQL' does not match inside 'MySQL' and 'Node.js' is not read as 'JS'.
Ambiguous names (R, C, Excel) use case-sensitive, context-aware patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .preprocessing import Segment
from .utils import SkillEntry, Taxonomy

REQUIRED_CUES = re.compile(
    r"\b(required|requires?|must[\s-]have|must|mandatory|essential|minimum|need to have)\b",
    re.IGNORECASE)
PREFERRED_CUES = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|good[\s-]to[\s-]have|plus|bonus|optional|desirable|"
    r"advantage|an asset|ideally)\b", re.IGNORECASE)

_LEFT_BOUNDARY = r"(?<![A-Za-z0-9_+#.])"
_RIGHT_BOUNDARY = r"(?![A-Za-z0-9_+#])"


@dataclass
class ExtractedSkill:
    """A skill found in the job description with the evidence used for prioritisation."""

    name: str
    category: str
    bucket: str
    mentions: int = 0          # number of sentences that mention the skill
    required_hits: int = 0     # sentences with 'required'/'must' wording or under a Requirements heading
    preferred_hits: int = 0    # sentences with 'preferred'/'nice to have' wording or under such a heading
    evidence: list = field(default_factory=list)


def _alias_regex(alias: str) -> str:
    """Escape an alias but let spaces and hyphens match either ('problem solving' = 'problem-solving')."""
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", alias.strip()) if p]
    return r"[\s\-]+".join(parts)


@lru_cache(maxsize=None)
def _compile(entry: SkillEntry) -> re.Pattern:
    labels = list(entry.aliases)
    if entry.match_name:
        labels.append(entry.name)
    alternatives = [_alias_regex(label) for label in sorted(set(labels), key=len, reverse=True)]
    alternatives.extend(entry.patterns)
    body = "(?:" + "|".join(alternatives) + ")"
    pattern = _LEFT_BOUNDARY + body + _RIGHT_BOUNDARY
    if entry.not_followed_by:
        pattern += f"(?!{entry.not_followed_by})"
    return re.compile(pattern, 0 if entry.case_sensitive else re.IGNORECASE)


def sentence_signal(segment: Segment) -> str:
    """Classify a segment as 'required', 'preferred' or 'general'.

    Wording inside the sentence wins over the section heading. If a sentence contains both kinds
    of wording it is treated as 'general' because the cue cannot be tied to one skill.
    """
    has_required = bool(REQUIRED_CUES.search(segment.text))
    has_preferred = bool(PREFERRED_CUES.search(segment.text))
    if has_required and not has_preferred:
        return "required"
    if has_preferred and not has_required:
        return "preferred"
    if has_required and has_preferred:
        return "general"
    return segment.section


def extract_skills(segments: list, taxonomy: Taxonomy) -> list:
    """Find taxonomy skills in the segments. Returns de-duplicated ExtractedSkill objects
    in order of first appearance."""
    found: dict = {}
    for segment in segments:
        signal = sentence_signal(segment)
        for entry in taxonomy.skills.values():
            if not _compile(entry).search(segment.text):
                continue
            skill = found.get(entry.name)
            if skill is None:
                skill = ExtractedSkill(name=entry.name, category=entry.category, bucket=entry.bucket)
                found[entry.name] = skill
            skill.mentions += 1
            if signal == "required":
                skill.required_hits += 1
            elif signal == "preferred":
                skill.preferred_hits += 1
            if len(skill.evidence) < 3:
                text = segment.text if len(segment.text) <= 200 else segment.text[:197] + "..."
                skill.evidence.append(text)
    return list(found.values())
