import importlib
import sys
from unittest.mock import patch
import sqlalchemy


def test_database_falls_back_to_sqlite_when_connection_url_is_unavailable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    original_create_engine = sqlalchemy.create_engine

    def broken_create_engine(url, *args, **kwargs):
        if url.startswith("sqlite"):
            return original_create_engine(url, *args, **kwargs)
        raise RuntimeError("simulated connection failure")

    sys.modules.pop("database", None)
    with patch("sqlalchemy.create_engine", side_effect=broken_create_engine):
        database = importlib.import_module("database")

    assert str(database.engine.url).startswith("sqlite")
    assert str(database.engine.url).endswith("jobprepmate.db")
