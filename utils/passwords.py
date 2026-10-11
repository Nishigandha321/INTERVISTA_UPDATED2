"""Password hashing helpers shared by authentication and recovery tools."""

import logging

from passlib.context import CryptContext


logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["pbkdf2_sha512", "bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash the complete password using the current application scheme."""
    return pwd_context.hash(password)


def _is_bcrypt_hash(hashed: str) -> bool:
    return isinstance(hashed, str) and hashed.startswith("$2")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify current PBKDF2 hashes and legacy bcrypt hashes."""
    try:
        if _is_bcrypt_hash(hashed):
            plain = plain[:72]
        return pwd_context.verify(plain, hashed)
    except Exception as exc:
        # Avoid including submitted credentials or stored hashes in diagnostics.
        logger.warning("Password verification failed (%s)", type(exc).__name__)
        return False
