"""JobLens AI - Streamlit dashboard.

Run with:  streamlit run app.py
All analysis logic lives in src/; this file only handles input, validation messages and display.
"""
from __future__ import annotations

import html
import logging

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:  # charts are optional; tables and cards still work
    px = None

from src.categorization import REQUIREMENT_BUCKETS
from src.pipeline import AnalysisResult, analyze
from src.preprocessing import MAX_CHARS, MIN_WORDS, InputError
from src.scoring import (HIGH_MENTION_THRESHOLD, MAX_PLAN_SKILLS, MODERATE_MATCH_PCT, PRIORITY_ORDER,
                         PRIORITY_WEIGHTS, STRONG_MATCH_PCT)
from src.utils import TaxonomyError, list_samples, load_taxonomy, read_sample

logger = logging.getLogger("joblens")

st.set_page_config(page_title="JobLens AI", page_icon="\U0001F50D", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 2rem;}
    .jl-chip {display:inline-block; margin:2px 6px 4px 0; padding:3px 11px; border-radius:14px; font-size:0.86rem;}
    .jl-sub {opacity:0.75; margin-top:-0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

CHIP_COLORS = {"match": "46,160,67", "missing": "218,54,51", "extra": "56,139,253", "neutral": "128,128,128"}
PRIORITY_COLORS = {"HIGH": "#d1453b", "MEDIUM": "#d99a1c", "LOW": "#6b8ea8"}


@st.cache_resource(show_spinner=False)
def get_taxonomy():
    return load_taxonomy()


def chips(items: list, kind: str) -> str:
    """HTML chips for a list of skill names."""
    if not items:
        return "<em>None</em>"
    rgb = CHIP_COLORS[kind]
    style = f"background:rgba({rgb},0.14); border:1px solid rgba({rgb},0.55);"
    return "".join(f"<span class='jl-chip' style='{style}'>{html.escape(str(i))}</span>" for i in items)


def load_sample_callback(path) -> None:
    try:
        st.session_state["jd_text"] = read_sample(path)
        st.session_state.pop("sample_error", None)
    except OSError as exc:
        st.session_state["sample_error"] = str(exc)


# ---------------------------------------------------------------- header
st.title("JobLens AI")
st.markdown("<p class='jl-sub'>Understand the Job. Identify Your Gap. Prepare Smarter.</p>",
            unsafe_allow_html=True)

try:
    taxonomy = get_taxonomy()
except TaxonomyError as exc:
    st.error(f"Could not load the skill taxonomy. {exc}")
    st.stop()
except Exception:  # noqa: BLE001 - never show a raw stack trace to users
    logger.exception("Unexpected error while loading the taxonomy")
    st.error("Something went wrong while loading the skill taxonomy. Check data/skills.json.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Paste a job description.\n"
        "2. Pick the skills you already have.\n"
        "3. Read the gap, the priorities and the plan."
    )
    samples = list_samples()
    if samples:
        st.subheader("Try a sample")
        label = st.selectbox("Sample job description", list(samples), key="sample_choice")
        st.button("Load sample", on_click=load_sample_callback, args=(samples[label],))
        if st.session_state.get("sample_error"):
            st.warning(st.session_state["sample_error"])
    st.divider()
    st.caption(
        "Skill detection is dictionary-based (data/skills.json). The match score is a "
        "skill-compatibility score, not a model accuracy. Nothing leaves your computer."
    )
    st.caption(f"Taxonomy: {len(taxonomy.skills)} skills in {len(taxonomy.category_buckets)} categories.")

# ---------------------------------------------------------------- inputs
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("1. Job description")
    jd_text = st.text_area(
        "Paste the full job description",
        key="jd_text",
        height=320,
        placeholder="Paste the whole posting: responsibilities, requirements, preferred skills...",
        help=f"At least {MIN_WORDS} words and under {MAX_CHARS:,} characters.",
    )
with right:
    st.subheader("2. Your skills")
    selected = st.multiselect(
        "Search and select skills you already have",
        options=taxonomy.names(),
        format_func=lambda n: f"{n}  ({taxonomy.category_of(n)})",
        key="profile_skills",
    )
    custom_text = st.text_input(
        "Add other skills (comma separated)",
        key="custom_skills",
        placeholder="e.g. python programming, Figma, statistics",
        help="Entries are normalised, so 'PYTHON' and 'Python programming' count as Python.",
    )

if st.button("Analyze", type="primary"):
    try:
        with st.spinner("Analysing..."):
            st.session_state["result"] = analyze(jd_text, [*selected, custom_text], taxonomy)
    except InputError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during analysis")
        st.session_state.pop("result", None)
        st.error("Something unexpected went wrong during the analysis. "
                 "Please check the job description text and try again.")

result: AnalysisResult | None = st.session_state.get("result")
if result is None:
    st.info("Paste a job description, choose your skills and click **Analyze**.")
    st.stop()


# ---------------------------------------------------------------- results
def render_overview(r: AnalysisResult) -> None:
    for message in r.warnings:
        st.warning(message)
    if not r.job_skills:
        return

    st.caption(f"Analysed {r.word_count} words in {len(r.segments)} sentences/bullets; "
               f"{r.score.total} skills detected.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total skills", r.score.total)
    c2.metric("Matched", r.score.matched)
    c3.metric("Missing", r.score.missing)
    c4.metric("Match score", f"{r.score.plain_pct:.0f}%", help="Matched skills / total job skills. Every skill counts equally.")
    c5.metric("Weighted score", f"{r.score.weighted_pct:.0f}%",
              help="Same idea, but HIGH priority skills count 3, MEDIUM 2 and LOW 1.")
    st.progress(min(max(r.score.weighted_pct / 100, 0.0), 1.0), text=f"{r.score.label} (weighted score)")

    if r.similarity is not None:
        st.metric("Related-skill similarity (TF-IDF)", f"{r.similarity * 100:.0f}%",
                  help="Supplementary signal. Compares the wording of your skills with the job's skills "
                       "and rewards related skills in the same category. It is not part of the match score.")

    chart_a, chart_b = st.columns(2)
    if px is None:
        st.info("Install plotly (pip install plotly) to see charts.")
        return
    donut_df = pd.DataFrame({"Status": ["Matched", "Missing"], "Skills": [r.score.matched, r.score.missing]})
    fig = px.pie(donut_df, names="Status", values="Skills", hole=0.55, color="Status",
                 color_discrete_map={"Matched": "#2ea043", "Missing": "#da3633"}, title="Matched vs missing")
    fig.update_layout(height=320, margin=dict(t=50, b=10, l=10, r=10))
    chart_a.plotly_chart(fig)

    rows = [{"Category": s.category, "Status": "Matched" if s.name in r.match.matched else "Missing"}
            for s in r.job_skills]
    cat_df = pd.DataFrame(rows).value_counts().reset_index(name="Skills")
    fig2 = px.bar(cat_df, x="Skills", y="Category", color="Status", orientation="h",
                  color_discrete_map={"Matched": "#2ea043", "Missing": "#da3633"}, title="Skills by category")
    fig2.update_layout(height=320, margin=dict(t=50, b=10, l=10, r=10), yaxis_title=None)
    chart_b.plotly_chart(fig2)


def render_skills(r: AnalysisResult) -> None:
    if not r.job_skills:
        st.info("No skills were detected in the job description.")
        return
    st.markdown(f"**Matched skills ({len(r.match.matched)})**")
    st.markdown(chips(r.match.matched, "match"), unsafe_allow_html=True)
    st.markdown(f"**Missing skills ({len(r.match.missing)})**")
    st.markdown(chips(r.match.missing, "missing"), unsafe_allow_html=True)
    st.markdown(f"**Additional skills you have ({len(r.match.additional)})**")
    st.markdown(chips(r.match.additional, "extra"), unsafe_allow_html=True)
    for typed, suggestion in r.profile.suggestions.items():
        st.caption(f"'{typed}' is not recognised. Did you mean '{suggestion}'?")
    if r.profile.ignored:
        st.caption("Ignored entries (empty or too long): " + ", ".join(r.profile.ignored))

    with st.expander("Where each job skill was found"):
        table = pd.DataFrame([{
            "Skill": s.name, "Category": s.category, "Sentences": s.mentions,
            "Evidence": " | ".join(s.evidence)} for s in r.job_skills])
        st.dataframe(table, hide_index=True)


def render_requirements(r: AnalysisResult) -> None:
    for bucket in REQUIREMENT_BUCKETS:
        items = r.requirements.get(bucket, [])
        with st.expander(f"{bucket} ({len(items)})", expanded=bucket in ("Technical Skills", "Tools & Technologies")):
            if not items:
                st.write("Nothing detected in this category.")
            elif bucket in ("Technical Skills", "Soft Skills", "Tools & Technologies"):
                st.markdown(chips(items, "neutral"), unsafe_allow_html=True)
            else:
                for item in items:
                    st.markdown(f"- {item}")
    if r.frequent_unknown_terms:
        with st.expander("Frequent terms that are not in the taxonomy"):
            st.caption("These words appear repeatedly but are not known skills. If one is a skill, "
                       "add it to data/skills.json.")
            st.dataframe(pd.DataFrame(r.frequent_unknown_terms, columns=["Term", "Count"]), hide_index=True)


def render_preparation(r: AnalysisResult) -> None:
    if not r.match.missing:
        st.success("No missing skills. Nothing to prepare for this job.")
        return
    cols = st.columns(3)
    for col, level in zip(cols, PRIORITY_ORDER):
        names = [n for n in r.match.missing if r.priorities[n].priority == level]
        col.markdown(f"<span style='color:{PRIORITY_COLORS[level]}; font-weight:600'>{level} ({len(names)})</span>",
                     unsafe_allow_html=True)
        col.markdown(chips(names, "missing" if level == "HIGH" else "neutral"), unsafe_allow_html=True)

    table = pd.DataFrame([{
        "Skill": n, "Category": r.priorities[n].category, "Priority": r.priorities[n].priority,
        "Weight": r.priorities[n].weight, "Why": r.priorities[n].reason} for n in r.match.missing])
    table["_o"] = table["Priority"].map(PRIORITY_ORDER.index)
    st.dataframe(table.sort_values("_o", kind="stable").drop(columns="_o"), hide_index=True)

    st.subheader("Preparation plan")
    st.caption("Rule-based planner: fixed rules turn priorities into a schedule. It is not generative AI.")
    for step in r.plan.steps:
        with st.container(border=True):
            tag = f" ({step.priority})" if step.priority in PRIORITY_COLORS else ""
            st.markdown(f"**{step.days}: {step.focus}**{tag}")
            for task in step.tasks:
                st.markdown(f"- {task}")
    if r.plan.deferred:
        st.caption("Not scheduled yet (lower priority): " + ", ".join(r.plan.deferred))


def render_method() -> None:
    st.markdown(
        f"""
**Skill detection.** Each skill in `data/skills.json` is matched in the text through its name and aliases
(dictionary + regular expressions). This is rule-based NLP, not a trained model, so skills missing from the
taxonomy are not detected.

**Match score.** `matched skills / total job skills x 100`. The weighted score uses the same idea but each skill
counts HIGH = {PRIORITY_WEIGHTS['HIGH']}, MEDIUM = {PRIORITY_WEIGHTS['MEDIUM']}, LOW = {PRIORITY_WEIGHTS['LOW']}.
Labels: Strong match at {STRONG_MATCH_PCT}% or more, Moderate at {MODERATE_MATCH_PCT}% or more, otherwise Needs preparation.
These weights and thresholds are design choices, not measured values. This is a compatibility score, not model accuracy.

**Priority of missing skills.**
- HIGH: a sentence mentioning the skill says required / must / mandatory / essential / minimum, or the skill sits under a
  Requirements or Qualifications heading, or it appears in {HIGH_MENTION_THRESHOLD} or more sentences.
- LOW: it only appears once, in a preferred / nice to have / plus context.
- MEDIUM: everything else.

**TF-IDF similarity.** The skills of the job and your skills are written as short documents (name, aliases and category).
TF-IDF weights rare words more than common ones and cosine similarity compares the two documents. It adds one thing the
exact match cannot: skills from the same category (for example Pandas and NumPy) count as related. It is shown separately.

**Preparation plan.** HIGH skills get 2 days, MEDIUM and LOW get 1 day (up to {MAX_PLAN_SKILLS} skills), followed by a mini project.
        """
    )


tabs = st.tabs(["Overview", "Skills", "Requirements", "Preparation", "How it is calculated"])
with tabs[0]:
    render_overview(result)
with tabs[1]:
    render_skills(result)
with tabs[2]:
    render_requirements(result)
with tabs[3]:
    render_preparation(result)
with tabs[4]:
    render_method()
