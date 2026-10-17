import json

import pytest

from src.utils import TaxonomyError, load_taxonomy


def test_default_taxonomy_loads(taxonomy):
    assert len(taxonomy.skills) > 50
    assert taxonomy.alias_lookup["python"] == "Python"


def test_missing_file(tmp_path):
    with pytest.raises(TaxonomyError, match="not found"):
        load_taxonomy(tmp_path / "nope.json")


def test_malformed_json(tmp_path):
    bad = tmp_path / "skills.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="not valid JSON"):
        load_taxonomy(bad)


def test_missing_sections(tmp_path):
    bad = tmp_path / "skills.json"
    bad.write_text(json.dumps({"skills": []}), encoding="utf-8")
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_conflicting_alias(tmp_path):
    doc = {"category_buckets": {"A": "Technical Skills"},
           "skills": [{"name": "X", "category": "A", "aliases": ["same"]},
                      {"name": "Y", "category": "A", "aliases": ["same"]}]}
    bad = tmp_path / "skills.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(TaxonomyError, match="unique"):
        load_taxonomy(bad)
