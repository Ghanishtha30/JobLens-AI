from src.preprocessing import split_into_segments
from src.skill_extraction import extract_skills


def names(text, taxonomy):
    return [s.name for s in extract_skills(split_into_segments(text), taxonomy)]


def test_java_is_not_javascript(taxonomy):
    assert names("Experience with JavaScript and React is needed here.", taxonomy) == ["JavaScript", "React"]
    assert "Java" not in names("We use JavaScript every day.", taxonomy)


def test_nodejs_is_not_js_and_sql_not_in_mysql(taxonomy):
    found = names("We build services in Node.js with MySQL storage.", taxonomy)
    assert "Node.js" in found and "MySQL" in found
    assert "JavaScript" not in found and "SQL" not in found


def test_cpp_and_csharp_survive(taxonomy):
    found = names("Knowledge of C++ and C# is useful for this position.", taxonomy)
    assert "C++" in found and "C#" in found and "C" not in found


def test_r_and_c_need_context(taxonomy):
    assert "R" not in names("Our R&D team works on new products and ideas daily.", taxonomy)
    assert "R" in names("Skills: Python, R, SQL and plenty of curiosity about data.", taxonomy)
    assert "C" in names("Languages such as Java, C or Python are all welcome here.", taxonomy)
    assert "C" not in names("You will report to the C-level management of the company.", taxonomy)


def test_excel_verb_is_ignored(taxonomy):
    assert "Excel" not in names("You will excel in a fast moving team environment here.", taxonomy)
    assert "Excel" in names("Proficiency in Microsoft Excel and reporting is expected.", taxonomy)


def test_alias_and_hyphen_variants(taxonomy):
    found = names("Strong problem-solving, sklearn and Power BI experience is needed.", taxonomy)
    assert {"Problem Solving", "Scikit-learn", "Power BI"} <= set(found)


def test_required_and_preferred_signals(taxonomy):
    text = "Requirements:\n- Python is needed daily.\nNice to have:\n- Docker exposure for deployments."
    skills = {s.name: s for s in extract_skills(split_into_segments(text), taxonomy)}
    assert skills["Python"].required_hits == 1
    assert skills["Docker"].preferred_hits == 1 and skills["Docker"].required_hits == 0
