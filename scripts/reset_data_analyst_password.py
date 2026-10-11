"""Interactive, no-echo password reset for the verified Data Analyst account."""

from getpass import getpass
from pathlib import Path
import sys

# When launched as ``python scripts/reset_data_analyst_password.py``, Python
# places ``scripts`` (rather than the project root) on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import SessionLocal
from services.data_analyst_password_reset import (
    TARGET_USER_ID,
    TARGET_USERNAME,
    PasswordResetError,
    reset_data_analyst_password,
)


def main() -> int:
    print(f"Target account: user ID {TARGET_USER_ID}, username {TARGET_USERNAME}")
    confirmation = input(f"Type {TARGET_USERNAME} to confirm: ")
    if confirmation != TARGET_USERNAME:
        print("Confirmation did not match; no changes were made.")
        return 1

    new_password = getpass("New password (input hidden): ")
    repeated_password = getpass("Repeat new password (input hidden): ")
    if new_password != repeated_password:
        print("The entries did not match; no changes were made.")
        return 1

    db = SessionLocal()
    try:
        reset_data_analyst_password(db, new_password)
    except PasswordResetError as exc:
        print(str(exc))
        return 1
    except Exception:
        # Do not print exception details that could include SQL parameters.
        print("Password reset failed; no success was reported. Check server-side diagnostics.")
        return 1
    finally:
        db.close()

    print("Password reset completed for the verified account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
