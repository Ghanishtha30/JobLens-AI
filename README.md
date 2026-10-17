# JobLens AI

> Understand the Job. Identify Your Gap. Prepare Smarter.

## 1. Overview
JobLens AI is a Streamlit app for students and fresh graduates. You paste a job description and select the skills you already have. The app extracts the skills the job asks for, compares them with yours, calculates an explainable skill-match score, ranks the missing skills by priority and (optionally) builds a rule-based preparation plan. It runs fully on your computer, with no API keys and no paid services.

## 2. Problem statement
Freshers often struggle to read long job descriptions and to tell which skills matter most, which they already have and what to learn before applying. JobLens AI answers five questions: what skills does the job require, which do I have, which am I missing, how well do I match, and what should I learn first.

## 3. SDG alignment
- **SDG 8 (primary): Decent Work and Economic Growth.** Helps young job seekers understand employer expectations and prepare for work.
- **SDG 4 (secondary): Quality Education.** Turns real job requirements into a focused self-learning plan.

## 4. Objectives
1. Extract and categorise requirements from a pasted job description.
2. Compare them with a user's skill profile and report matched, missing and additional skills.
3. Give a transparent score and priority for each missing skill.
4. Present everything in a clear dashboard.

## 5. Features
- Job description input with validation (empty, too short, too long).
- Skill extraction from an editable taxonomy (`data/skills.json`).
- Seven requirement categories: Technical Skills, Soft Skills, Tools & Technologies, Education, Experience, Certifications, Other Requirements.
- User skill profile: searchable multiselect, free-text additions, normalisation of spellings and duplicates, "Did you mean...?" hints for typos.
- Matched / missing / additional skills; plain and weighted match score; TF-IDF similarity shown separately.
- HIGH / MEDIUM / LOW priority for missing skills, each with a stated reason.
- Rule-based preparation plan and mini-project suggestion.
- Plotly charts, metric cards, evidence table, and a "How it is calculated" tab.

## 6. Architecture
```
Job description -> validation -> segmentation (sentences + Requirements / Preferred sections)
   -> taxonomy skill extraction -> categorisation
User skills -> normalisation -> matching (matched / missing / additional)
   -> priority rules -> match score -> TF-IDF similarity -> preparation plan -> dashboard
```
`app.py` handles the UI only. `src/pipeline.py` runs the whole analysis, and the same function is used by the tests.

## 7. AI / NLP methodology
| Step | Method | Notes |
|---|---|---|
| Preprocessing | Rule-based text cleaning, sentence segmentation, heading detection | Keeps `C++`, `C#`, `Node.js`, `CI/CD`. Original case is kept because R, C and Excel need case-sensitive matching. |
| Skill extraction | Dictionary + alias matching with regular expressions | **Not a trained model.** Skills missing from `skills.json` are not detected. |
| Categorisation | Taxonomy categories plus regular-expression rules for education, experience and certifications | Rules are in `src/categorization.py`. |
| Match score | `matched / total x 100` (plain) and a weighted version | Weights: HIGH = 3, MEDIUM = 2, LOW = 1. These are design choices, not learned values. **It is a compatibility score, not model accuracy.** |
| Priority | Wording cues (`required`, `must`, `mandatory`, `preferred`, `nice to have`, ...), section headings, number of sentences mentioning the skill | HIGH: required signal or 3+ sentences. LOW: one mention in preferred context. MEDIUM: otherwise. |
| Text similarity | **TF-IDF + cosine similarity** (scikit-learn) | The job's skills and the user's skills become short documents (name, aliases, category). The vocabulary is learned from the whole taxonomy. It rewards related skills in the same category. It is supplementary and is not part of the match score. |
| Preparation plan | Fixed rules and templates | **Not generative AI.** HIGH skills get 2 days, MEDIUM/LOW 1 day, then a mini project. |

No LLM or external API is used, and there is no separate "agent" component: the planner already produces the preparation sequence and project idea.

## 8. Technology stack
Python 3.10+, Streamlit, Pandas, NumPy, scikit-learn, Plotly. Tests use pytest.

## 9. Dataset / source
No external dataset is used. The skill taxonomy (`data/skills.json`) was written by hand for this project, and the two files in `data/samples/` are short illustrative job descriptions written for demonstration, not real postings. If you later add a public dataset, record its exact name, URL, licence and number of records here after checking them yourself.

## 10. Installation
```bash
git clone https://github.com/<your-username>/JobLens-AI.git
cd JobLens-AI
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## 11. Running the application
```bash
streamlit run app.py
```

## 12. Usage
1. (Optional) pick a sample in the sidebar and click **Load sample**, or paste your own job description.
2. Select your skills and/or type extra ones, separated by commas.
3. Click **Analyze** and read the tabs: Overview, Skills, Requirements, Preparation, How it is calculated.

## 13. Testing
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
Covered: Data Analyst JD with Python/Pandas/Excel, Software Developer JD with Java/Git/SQL, a JD with no recognised skills, empty or tiny input, duplicate user skills, ambiguous skills (R, C, Java vs JavaScript, SQL vs MySQL, Node.js), required/preferred signals, and taxonomy file errors (missing file, malformed JSON, conflicting aliases).
Manual UI checks: see the five test scenarios in the project notes and compare with the expected behaviour.

## 14. Limitations
- Only skills present in `data/skills.json` are detected (73 at the time of writing); spelling variants not listed as aliases are missed.
- The priority and score weights are simple heuristics. The wording cues may misread unusual job descriptions.
- Short names (R, C) are matched only in list-like contexts, so some real mentions can be missed.
- TF-IDF similarity depends on the taxonomy categories and is only a rough signal.
- English text only. No resume parsing, no live job search.

## 15. Future scope
Larger or verified taxonomies, optional embeddings for synonym matching, resume upload, saving analyses, and an optional LLM-written explanation (the core app must keep working without it).

## 16. Screenshots
Add your own screenshots after running the app. Nothing below is generated.

| Screen | File |
|---|---|
| Input screen | `assets/screenshots/01_input.png` _(to be added)_ |
| Overview dashboard | `assets/screenshots/02_overview.png` _(to be added)_ |
| Skills tab | `assets/screenshots/03_skills.png` _(to be added)_ |
| Preparation tab | `assets/screenshots/04_preparation.png` _(to be added)_ |

## 17. Project structure
```
JobLens-AI/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
├── data/
│   ├── skills.json            # skill taxonomy (edit to add skills)
│   └── samples/               # two illustrative job descriptions
├── src/
│   ├── preprocessing.py
│   ├── skill_extraction.py
│   ├── categorization.py
│   ├── matching.py
│   ├── scoring.py
│   ├── pipeline.py
│   └── utils.py
├── tests/
└── assets/screenshots/
```

## 18. Adding a skill
Append an entry to `data/skills.json`:
```json
{ "name": "Figma", "category": "Web", "aliases": ["figma design"] }
```
Use `"case_sensitive": true` for ambiguous names. Aliases must be unique across skills.

## 19. Licence / attribution
Add your preferred licence (for example MIT) as a `LICENSE` file. Libraries used: Streamlit, Pandas, NumPy, scikit-learn and Plotly, each under its own open-source licence. Built as part of the IBM SkillsBuild – AICTE – BharatCares (CSRBOX) Machine Learning & Applied AI Internship 2026.
