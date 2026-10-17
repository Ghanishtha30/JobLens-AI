"""The five required scenarios, run through the same analyze() function the app uses."""
import pytest

from src.pipeline import analyze
from src.preprocessing import InputError


def test1_data_analyst_python_pandas_excel(taxonomy, data_analyst_jd):
    r = analyze(data_analyst_jd, ["Python", "Pandas", "Excel"], taxonomy)
    assert r.match.matched == ["Python", "Pandas", "Excel"]
    for skill in ("SQL", "Power BI", "Data Visualization", "Tableau", "Machine Learning"):
        assert skill in r.match.missing
    assert r.score.total == len(r.job_skills) == 11
    assert r.score.matched == 3
    assert r.score.plain_pct == 27.3
    assert r.priorities["SQL"].priority == "HIGH"       # "Python and SQL are required"
    assert r.priorities["Tableau"].priority == "LOW"    # only under "Nice to have"
    assert r.priorities["Power BI"].priority == "MEDIUM"  # general + preferred mentions


def test2_software_developer_java_git_sql(taxonomy, developer_jd):
    r = analyze(developer_jd, ["Java", "Git", "SQL"], taxonomy)
    assert r.match.matched == ["Java", "SQL", "Git"]
    assert {"Docker", "AWS", "REST APIs", "Spring Boot", "GitHub"} <= set(r.match.missing)
    assert r.priorities["Git"].priority == "HIGH"        # "mandatory"
    assert r.priorities["Docker"].priority == "LOW"      # "Preferred qualifications"
    assert r.score.matched == 3 and r.score.total == 9


def test3_no_recognised_skills(taxonomy):
    text = ("We are looking for a friendly person to join our office. The role involves answering "
            "phones and greeting visitors every day. Applicants should be punctual and polite.")
    r = analyze(text, ["Python"], taxonomy)
    assert r.job_skills == []
    assert r.score.total == 0 and r.score.plain_pct == 0.0
    assert any("No skills" in w for w in r.warnings)
    assert r.plan.steps == []


@pytest.mark.parametrize("text", ["", "   \n  ", None, "too short"])
def test4_empty_or_tiny_input_raises(taxonomy, text):
    with pytest.raises(InputError):
        analyze(text, ["Python"], taxonomy)


def test5_duplicate_user_skills(taxonomy, data_analyst_jd):
    r = analyze(data_analyst_jd, ["python", "Python", "PYTHON", "Python programming"], taxonomy)
    assert r.profile.taxonomy_skills == ["Python"]
    assert r.profile.duplicates_removed == 3
    assert r.match.matched == ["Python"]


def test_empty_profile_everything_missing(taxonomy, data_analyst_jd):
    r = analyze(data_analyst_jd, [], taxonomy)
    assert r.match.matched == [] and r.score.plain_pct == 0.0
    assert any("empty" in w for w in r.warnings)


def test_unknown_user_skill_is_kept_as_additional(taxonomy, data_analyst_jd):
    r = analyze(data_analyst_jd, ["Figma", "pyhton"], taxonomy)
    assert "Figma" in r.match.additional
    assert r.profile.suggestions.get("pyhton") == "Python"
