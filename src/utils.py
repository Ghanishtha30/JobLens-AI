"""Shared helpers: project paths, skill-taxonomy loading/validation, small text utilities.

The taxonomy lives in data/skills.json so it can be expanded without touching code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SKILLS_PATH = DATA_DIR / "skills.json"
SAMPLES_DIR = DATA_DIR / "samples"


class TaxonomyError(Exception):
    """Raised when skills.json is missing, malformed or inconsistent.

    The message is written for the end user and is safe to display.
    """


@dataclass(frozen=True)
class SkillEntry:
    """One skill in the taxonomy."""

    name: str
    category: str
    bucket: str
    aliases: tuple = ()
    case_sensitive: bool = False
    match_name: bool = True
    patterns: tuple = ()
    not_followed_by: str = ""


@dataclass
class Taxonomy:
    """The loaded skill taxonomy plus a lookup table for normalizing text."""

    skills: dict           # canonical name -> SkillEntry
    alias_lookup: dict     # normalized alias/name -> canonical name
    category_buckets: dict  # category -> requirement bucket

    def names(self) -> list:
        """Canonical skill names ordered by category, then alphabetically."""
        order = list(self.category_buckets)
        entries = sorted(
            self.skills.values(),
            key=lambda e: (order.index(e.category) if e.category in order else 99, e.name.lower()),
        )
        return [e.name for e in entries]

    def category_of(self, name: str) -> str:
        entry = self.skills.get(name)
        return entry.category if entry else "Other"


def normalize_key(text: str) -> str:
    """Lowercase, collapse whitespace and trim edge punctuation.

    Characters such as '+', '#' and inner dots are kept so 'C++', 'C#' and 'Node.js' survive.
    """
    cleaned = re.sub(r"\s+", " ", str(text).replace("\u2019", "'")).strip().lower()
    return cleaned.strip(" .,;:!?\"'()[]{}-_/")


def load_taxonomy(path: Path = SKILLS_PATH) -> Taxonomy:
    """Load and validate the skill taxonomy.

    Raises TaxonomyError with a friendly message for a missing file, malformed JSON,
    missing fields, invalid regular expressions or conflicting aliases.
    """
    path = Path(path)
    if not path.exists():
        raise TaxonomyError(
            f"The skill file '{path.name}' was not found. Make sure the 'data' folder "
            "is next to app.py and contains skills.json."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaxonomyError(
            f"'{path.name}' is not valid JSON (line {exc.lineno}, column {exc.colno}). "
            "Fix the file or restore the original."
        ) from exc
    except OSError as exc:
        raise TaxonomyError(f"'{path.name}' could not be read: {exc.strerror or exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("skills"), list) \
            or not isinstance(raw.get("category_buckets"), dict):
        raise TaxonomyError(
            f"'{path.name}' must contain a 'category_buckets' object and a 'skills' list."
        )

    category_buckets = {str(k): str(v) for k, v in raw["category_buckets"].items()}
    skills: dict = {}
    alias_lookup: dict = {}

    for index, item in enumerate(raw["skills"], start=1):
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            raise TaxonomyError(f"Skill #{index} in '{path.name}' has no 'name'.")
        name = str(item["name"]).strip()
        category = str(item.get("category", "")).strip()
        if category not in category_buckets:
            raise TaxonomyError(
                f"Skill '{name}' uses category '{category}', which is not listed in 'category_buckets'."
            )
        if name in skills:
            raise TaxonomyError(f"Skill '{name}' appears more than once in '{path.name}'.")

        patterns = tuple(str(p) for p in item.get("patterns", []))
        for pattern in patterns + (str(item.get("not_followed_by", "")),):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise TaxonomyError(f"Skill '{name}' has an invalid regular expression: {exc}") from exc

        entry = SkillEntry(
            name=name,
            category=category,
            bucket=category_buckets[category],
            aliases=tuple(str(a).strip() for a in item.get("aliases", []) if str(a).strip()),
            case_sensitive=bool(item.get("case_sensitive", False)),
            match_name=bool(item.get("match_name", True)),
            patterns=patterns,
            not_followed_by=str(item.get("not_followed_by", "")),
        )
        skills[name] = entry

        for label in (name, *entry.aliases):
            key = normalize_key(label)
            if key in alias_lookup and alias_lookup[key] != name:
                raise TaxonomyError(
                    f"'{label}' is used for both '{alias_lookup[key]}' and '{name}'. Aliases must be unique."
                )
            alias_lookup[key] = name

    if not skills:
        raise TaxonomyError(f"'{path.name}' does not contain any skills.")
    return Taxonomy(skills=skills, alias_lookup=alias_lookup, category_buckets=category_buckets)


def list_samples(directory: Path = SAMPLES_DIR) -> dict:
    """Return {label: path} for the bundled sample job descriptions (empty if none)."""
    if not Path(directory).is_dir():
        return {}
    return {
        p.stem.replace("_", " ").title(): p
        for p in sorted(Path(directory).glob("*.txt"))
    }


def read_sample(path: Path) -> str:
    """Read a sample job description, raising a friendly OSError message on failure."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not read sample file '{Path(path).name}'.") from exc
