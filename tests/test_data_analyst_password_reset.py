"""Focused safety tests for the ID- and username-pinned recovery operation."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from models import User
from services.data_analyst_password_reset import (
    PasswordResetError,
    reset_data_analyst_password,
)
from utils.passwords import hash_password, verify_password


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        User(id=12, username="demo_data_analyst_2026", password=hash_password("old-target-password")),
        User(id=13, username="unrelated-account", password=hash_password("other-account-password")),
    ])
    session.commit()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_reset_updates_only_verified_account_and_password_hash(db_session):
    other_before = db_session.query(User).filter_by(id=13).one().password

    reset_data_analyst_password(db_session, "new-secure-test-password")

    target = db_session.query(User).filter_by(id=12).one()
    other = db_session.query(User).filter_by(id=13).one()
    assert target.username == "demo_data_analyst_2026"
    assert verify_password("new-secure-test-password", target.password)
    assert target.password != "new-secure-test-password"
    assert other.password == other_before
    assert verify_password("other-account-password", other.password)


@pytest.mark.parametrize("new_password", ["too-short", ""])
def test_reset_rejects_invalid_password_without_changing_accounts(db_session, new_password):
    hashes_before = {row.id: row.password for row in db_session.query(User).all()}

    with pytest.raises(PasswordResetError):
        reset_data_analyst_password(db_session, new_password)

    assert {row.id: row.password for row in db_session.query(User).all()} == hashes_before


def test_reset_rejects_missing_verified_account_without_changing_other_user(db_session):
    target = db_session.query(User).filter_by(id=12).one()
    target.username = "unexpected-username"
    db_session.commit()
    hashes_before = {row.id: row.password for row in db_session.query(User).all()}

    with pytest.raises(PasswordResetError, match="verified account was not found"):
        reset_data_analyst_password(db_session, "new-secure-test-password")

    assert {row.id: row.password for row in db_session.query(User).all()} == hashes_before
    assert verify_password(
        "other-account-password", db_session.query(User).filter_by(id=13).one().password
    )
