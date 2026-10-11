import importlib
import sys
from unittest.mock import patch
import sqlalchemy
import pytest


def test_database_does_not_fall_back_to_sqlite_when_primary_engine_creation_fails(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://invalid.invalid/nonexistent")

    def broken_create_engine(url, *args, **kwargs):
        if url.startswith("sqlite"):
            return original_create_engine(url, *args, **kwargs)
        raise RuntimeError("simulated connection failure")

    sys.modules.pop("database", None)
    with patch("sqlalchemy.create_engine", side_effect=broken_create_engine):
        with pytest.raises(RuntimeError, match="simulated connection failure"):
            importlib.import_module("database")


def test_database_does_not_fall_back_when_primary_schema_initialization_fails(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://invalid.invalid/nonexistent")
    sys.modules.pop("database", None)
    database = importlib.import_module("database")

    with patch.object(database.Base.metadata, "create_all", side_effect=RuntimeError("schema unavailable")):
        with pytest.raises(RuntimeError, match="schema unavailable"):
            database.initialize_database()

    assert str(database.engine.url).startswith("postgresql")
