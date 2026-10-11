"""Narrowly scoped recovery for the verified Data Analyst account."""

from sqlalchemy.orm import Session

from models import User
from utils.passwords import hash_password


TARGET_USER_ID = 12
TARGET_USERNAME = "demo_data_analyst_2026"
MIN_PASSWORD_LENGTH = 12


class PasswordResetError(ValueError):
    """Raised when the requested recovery does not match the verified account."""


def reset_data_analyst_password(db: Session, new_password: str) -> None:
    """Change only the password for the exact, pre-verified account.

    This operation intentionally cannot be pointed at another user. It requires
    an exact ID and username match and commits no other field or record changes.
    """
    if not isinstance(new_password, str) or len(new_password) < MIN_PASSWORD_LENGTH:
        raise PasswordResetError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )

    user = (
        db.query(User)
        .filter(User.id == TARGET_USER_ID, User.username == TARGET_USERNAME)
        .one_or_none()
    )
    if user is None:
        raise PasswordResetError("The verified account was not found; no changes were made.")

    user.password = hash_password(new_password)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
