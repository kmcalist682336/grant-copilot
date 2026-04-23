"""sys.path setup + shared fixtures."""
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def gazetteer_db():
    """Session-scoped SpatiaLite connection — geo_resolver/composer tests."""
    from scripts.chatbot.gazetteer_db import open_spatialite
    db = open_spatialite(REPO_ROOT / "data" / "geo" / "gazetteer.db")
    yield db
    db.close()


@pytest.fixture(scope="session")
def gold_entries():
    path = REPO_ROOT / "tests" / "fixtures" / "gold_queries.yaml"
    with path.open() as f:
        return yaml.safe_load(f)["entries"]


@pytest.fixture(scope="session")
def metadata_db():
    """Session-scoped read-only handle to the Tier-2 metadata FTS index."""
    from scripts.chatbot.metadata_search import open_search_index
    db = open_search_index(REPO_ROOT / "data" / "metadata" / "search_index.db")
    yield db
    db.close()
