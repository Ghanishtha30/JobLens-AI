import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_taxonomy  # noqa: E402


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="session")
def data_analyst_jd():
    return (ROOT / "data" / "samples" / "data_analyst.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def developer_jd():
    return (ROOT / "data" / "samples" / "software_developer.txt").read_text(encoding="utf-8")
