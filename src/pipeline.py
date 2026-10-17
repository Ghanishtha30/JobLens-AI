"""One function that runs the whole analysis, so the UI and the tests share the same code path."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .categorization import categorize_requirements
from .matching import MatchResult, NormalizedProfile, match_skills, normalize_user_skills, tfidf_similarity
from .preprocessing import (Segment, clean_text, split_into_segments, top_terms, validate_job_description)
from .scoring import PrepPlan, ScoreBreakdown, SkillPriority, build_prep_plan, compute_score, prioritize
from .skill_extraction import ExtractedSkill, extract_skills
from .utils import Taxonomy


@dataclass
class AnalysisResult:
    word_count: int
    segments: list
    job_skills: list                 # list[ExtractedSkill]
    requirements: dict
    profile: NormalizedProfile
    match: MatchResult
    priorities: dict                 # name -> SkillPriority
    score: ScoreBreakdown
    similarity: Optional[float]
    plan: PrepPlan
    frequent_unknown_terms: list
    warnings: list = field(default_factory=list)

    @property
    def job_skill_names(self) -> list:
        return [s.name for s in self.job_skills]


def analyze(jd_text: str, user_skills, taxonomy: Taxonomy) -> AnalysisResult:
    """Run extraction, matching, scoring and planning.

    Raises InputError (from preprocessing) for an empty or too-short job description.
    Soft problems (no skills found, empty profile, unknown skills) become `warnings`.
    """
    text = validate_job_description(jd_text)
    segments = split_into_segments(text)
    job_skills = extract_skills(segments, taxonomy)
    names = [s.name for s in job_skills]

    profile = normalize_user_skills(user_skills, taxonomy)
    match = match_skills(names, profile.all_skills)
    priorities = prioritize(job_skills)
    score = compute_score(names, match.matched, priorities)
    similarity = tfidf_similarity(names, profile.all_skills, taxonomy) if names else 0.0
    plan = build_prep_plan([priorities[n] for n in match.missing])
    requirements = categorize_requirements(job_skills, segments)

    known_words = set(taxonomy.alias_lookup)
    for key in list(taxonomy.alias_lookup):
        known_words.update(key.split())
    unknown_terms = top_terms(clean_text(text), known_words)

    warnings: list = []
    if not job_skills:
        warnings.append(
            "No skills from the taxonomy were found in this job description. "
            "It may use skills that are not in data/skills.json yet - see the frequent terms below."
        )
    if not profile.all_skills:
        warnings.append("Your skill profile is empty, so every required skill is shown as missing.")
    if profile.custom_skills:
        warnings.append(
            "Not in the taxonomy (kept as additional skills): " + ", ".join(profile.custom_skills) + "."
        )
    if profile.duplicates_removed:
        warnings.append(f"{profile.duplicates_removed} duplicate skill entr"
                        f"{'y was' if profile.duplicates_removed == 1 else 'ies were'} merged.")

    return AnalysisResult(
        word_count=len(text.split()), segments=segments, job_skills=job_skills,
        requirements=requirements, profile=profile, match=match, priorities=priorities,
        score=score, similarity=similarity, plan=plan,
        frequent_unknown_terms=unknown_terms, warnings=warnings,
    )
