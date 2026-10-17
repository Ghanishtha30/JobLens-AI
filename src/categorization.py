"""Rule-based categorisation of job requirements into seven buckets.

Skill buckets come from the taxonomy (category_buckets in skills.json). Education, Experience,
Certifications and Other are found with the transparent regular-expression rules below.
"""
from __future__ import annotations

import re

REQUIREMENT_BUCKETS = [
    "Technical Skills",
    "Soft Skills",
    "Tools & Technologies",
    "Education",
    "Experience",
    "Certifications",
    "Other Requirements",
]

_EDUCATION = re.compile(
    r"\b(b\.?\s?tech|m\.?\s?tech|b\.e\b|m\.e\b|bachelor'?s?|masters?|master'?s|master of|mca|bca|"
    r"b\.?\s?sc|m\.?\s?sc|ph\.?d|degree|diploma|graduates?|graduation)\b", re.IGNORECASE)
_EXPERIENCE = re.compile(
    r"\b(\d+\s*(?:\+|-|\u2013|to)?\s*\d*\s*(?:years?|yrs?)|freshers?|entry[\s-]level|internships?|"
    r"experience|experienced)\b", re.IGNORECASE)
_CERTIFICATION = re.compile(r"\b(certif\w*|licen[sc]e[sd]?)\b", re.IGNORECASE)
_OTHER_CUES = re.compile(
    r"\b(must|should|required|mandatory|responsible for|ability to|able to|willing|proficien\w*|"
    r"familiar\w*|knowledge of|understanding of|exposure to|working knowledge)\b", re.IGNORECASE)

_TEXT_RULES = [
    ("Education", _EDUCATION),
    ("Experience", _EXPERIENCE),
    ("Certifications", _CERTIFICATION),
]
_MAX_PER_BUCKET = 10


def _add(target: list, text: str) -> None:
    text = text if len(text) <= 240 else text[:237] + "..."
    if text not in target and len(target) < _MAX_PER_BUCKET:
        target.append(text)


def categorize_requirements(skills: list, segments: list) -> dict:
    """Return {bucket: [items]}. Skill buckets hold skill names; the others hold sentences."""
    result = {bucket: [] for bucket in REQUIREMENT_BUCKETS}
    for skill in skills:
        if skill.bucket in result:
            result[skill.bucket].append(skill.name)

    for segment in segments:
        matched = False
        for bucket, rule in _TEXT_RULES:
            if rule.search(segment.text):
                _add(result[bucket], segment.text)
                matched = True
        if not matched and _OTHER_CUES.search(segment.text):
            _add(result["Other Requirements"], segment.text)
    return result
