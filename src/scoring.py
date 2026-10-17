"""Priority rules, the explainable match score and the rule-based preparation planner.

Everything here is a fixed, documented rule. Nothing is learned from data, and the numbers
(weights, thresholds) are design choices, not measured values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .skill_extraction import ExtractedSkill

PRIORITY_ORDER = ("HIGH", "MEDIUM", "LOW")
PRIORITY_WEIGHTS = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
HIGH_MENTION_THRESHOLD = 3     # mentioned in this many sentences -> HIGH
STRONG_MATCH_PCT = 75
MODERATE_MATCH_PCT = 50
MAX_PLAN_SKILLS = 6


@dataclass
class SkillPriority:
    name: str
    category: str
    priority: str
    weight: int
    reason: str


@dataclass
class ScoreBreakdown:
    total: int
    matched: int
    missing: int
    plain_pct: float
    weighted_pct: float
    matched_weight: int
    total_weight: int
    label: str


@dataclass
class PlanStep:
    days: str
    focus: str
    priority: str
    tasks: list


@dataclass
class PrepPlan:
    steps: list = field(default_factory=list)
    project: str = ""
    deferred: list = field(default_factory=list)


def assign_priority(skill: ExtractedSkill) -> SkillPriority:
    """HIGH / MEDIUM / LOW from signals found in the job description (see README for rules)."""
    if skill.required_hits > 0:
        level = "HIGH"
        reason = (f"Found in {skill.required_hits} sentence(s) with 'required/must/mandatory' "
                  "wording or under a Requirements heading.")
    elif skill.mentions >= HIGH_MENTION_THRESHOLD:
        level = "HIGH"
        reason = f"Mentioned in {skill.mentions} different sentences."
    elif skill.preferred_hits == skill.mentions and skill.mentions <= 1:
        level = "LOW"
        reason = "Only mentioned in 'preferred / nice to have / plus' context."
    else:
        level = "MEDIUM"
        reason = (f"Mentioned in {skill.mentions} sentence(s) without an explicit 'required' signal.")
    return SkillPriority(skill.name, skill.category, level, PRIORITY_WEIGHTS[level], reason)


def prioritize(skills: list) -> dict:
    """Return {skill name: SkillPriority} for every skill extracted from the job description."""
    return {s.name: assign_priority(s) for s in skills}


def score_label(pct: float) -> str:
    if pct >= STRONG_MATCH_PCT:
        return "Strong match"
    if pct >= MODERATE_MATCH_PCT:
        return "Moderate match"
    return "Needs preparation"


def compute_score(job_skills: list, matched: list, priorities: dict) -> ScoreBreakdown:
    """Plain score = matched / total. Weighted score uses HIGH=3, MEDIUM=2, LOW=1."""
    total = len(job_skills)
    if total == 0:
        return ScoreBreakdown(0, 0, 0, 0.0, 0.0, 0, 0, "No skills detected")
    matched_set = {m.lower() for m in matched}
    total_weight = sum(priorities[name].weight for name in job_skills)
    matched_weight = sum(priorities[name].weight for name in job_skills if name.lower() in matched_set)
    plain = round(len(matched) / total * 100, 1)
    weighted = round(matched_weight / total_weight * 100, 1) if total_weight else 0.0
    return ScoreBreakdown(
        total=total, matched=len(matched), missing=total - len(matched),
        plain_pct=plain, weighted_pct=weighted,
        matched_weight=matched_weight, total_weight=total_weight,
        label=score_label(weighted),
    )


# (learn, practice, quick overview) per category; {s} is replaced by the skill name.
_TASK_TEMPLATES = {
    "Programming Languages": (
        "Learn core syntax, data types and control flow in {s}.",
        "Solve 5-10 small practice problems in {s} and keep the code on GitHub.",
        "Skim the basics of {s} and write one small program."),
    "Data Science / ML": (
        "Learn the core concepts and main functions of {s}.",
        "Apply {s} to a small public dataset and write down what you found.",
        "Read an introduction to {s} and run one worked example."),
    "Data / BI": (
        "Learn the fundamentals of {s} using a free tutorial or official documentation.",
        "Recreate a small report or chart in {s} from a sample dataset.",
        "Explore the basics of {s} and build one simple chart or table."),
    "Web": (
        "Learn the fundamentals of {s} from its official documentation.",
        "Build a small page or endpoint with {s}.",
        "Read the {s} quick-start and build a minimal example."),
    "Cloud / DevOps": (
        "Learn what {s} is used for and its core commands or concepts.",
        "Use {s} on a small personal project and document the steps.",
        "Read the {s} getting-started guide and try one example."),
    "Databases": (
        "Learn how to create tables and run basic queries in {s}.",
        "Load a small dataset into {s} and answer 5 questions with queries.",
        "Install {s} and run a handful of basic queries."),
    "Core CS": (
        "Revise the key concepts of {s} from your coursework or a free course.",
        "Practise interview-style questions on {s}.",
        "Revise the main ideas of {s} and answer a few practice questions."),
    "Soft Skills": (
        "Write down two real examples (project, club, class work) that show {s}.",
        "Practise explaining one example in 60-90 seconds using the situation-task-action-result structure.",
        "Prepare one short example that shows {s}."),
}
_DEFAULT_TEMPLATE = (
    "Learn the fundamentals of {s}.",
    "Practise {s} with a small exercise.",
    "Read an introduction to {s}.",
)


def _project_suggestion(focus: list) -> str:
    """Pick a mini-project idea from the categories of the skills being prepared."""
    categories = {p.category for p in focus}
    names = ", ".join(p.name for p in focus[:3])
    if categories & {"Data / BI", "Data Science / ML"}:
        return (f"Take a small public dataset, clean it, analyse it and present 3 findings using {names}. "
                "Publish the notebook or dashboard with a short README on GitHub.")
    if "Web" in categories:
        return f"Build a small web app or API that uses {names} and publish it on GitHub with a README."
    if "Cloud / DevOps" in categories:
        return f"Take an existing small project of yours and set it up using {names}. Document each step."
    return f"Build one small project that uses {names} together and document it in a GitHub README."


def build_prep_plan(missing: list) -> PrepPlan:
    """Rule-based planner: HIGH skills get 2 days, MEDIUM/LOW get 1 day, then a mini project.

    Not generative AI: the order and durations follow fixed rules from the priority levels.
    """
    ranked = sorted(missing, key=lambda p: PRIORITY_ORDER.index(p.priority))
    focus, deferred = ranked[:MAX_PLAN_SKILLS], ranked[MAX_PLAN_SKILLS:]
    plan = PrepPlan(deferred=[p.name for p in deferred])
    if not focus:
        return plan

    day = 1
    for item in focus:
        learn, practice, quick = _TASK_TEMPLATES.get(item.category, _DEFAULT_TEMPLATE)
        if item.priority == "HIGH":
            span, tasks = 2, [learn.format(s=item.name), practice.format(s=item.name)]
        else:
            span, tasks = 1, [quick.format(s=item.name)]
        days = f"Day {day}" if span == 1 else f"Day {day}\u2013{day + span - 1}"
        plan.steps.append(PlanStep(days, item.name, item.priority, tasks))
        day += span

    high_count = sum(1 for p in focus if p.priority == "HIGH")
    span = 2 if high_count >= 3 else 1
    plan.project = _project_suggestion([p for p in focus if p.priority == "HIGH"] or focus)
    days = f"Day {day}" if span == 1 else f"Day {day}\u2013{day + span - 1}"
    plan.steps.append(PlanStep(days, "Mini project", "-", [plan.project]))
    return plan
