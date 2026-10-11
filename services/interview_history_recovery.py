"""Pure reconciliation helpers for the one-time SQLite history recovery."""

from collections import Counter
from datetime import datetime
from typing import Any


class RecoveryConflict(RuntimeError):
    """A source session conflicts with already persisted destination data."""


def _timestamp_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat(sep=" ", timespec="microseconds")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None).isoformat(sep=" ", timespec="microseconds")
    except ValueError:
        return str(value)


def attempt_identity(row: dict) -> tuple:
    """Fields that must match exactly to count a saved answer as already copied."""
    return (
        row.get("role"),
        row.get("topic"),
        row.get("difficulty"),
        row.get("answer"),
        row.get("feedback"),
        _timestamp_key(row.get("timestamp")),
    )


def report_matches(source: dict, existing: dict) -> bool:
    return (
        existing.get("user_id") == source.get("user_id")
        and existing.get("role") == source.get("role")
        and float(existing.get("score") or 0) == float(source.get("score") or 0)
        and _timestamp_key(existing.get("date")) == _timestamp_key(source.get("date"))
        and existing.get("report_json") == source.get("report_json")
    )


def plan_session(
    source_report: dict,
    source_attempts: list[dict],
    existing_reports: list[dict],
    existing_attempts: list[dict],
    target_user_id: int,
) -> tuple[bool, list[dict]]:
    """Return (should_insert_report, missing_attempt_rows), without mutation.

    Existing data is accepted only when it is an exact subset of the source.
    Mismatches, extra destination answers, or foreign ownership stop recovery.
    """
    session_id = source_report.get("session_id")
    if not session_id:
        raise RecoveryConflict("Source report has no session ID.")
    if len(existing_reports) > 1:
        raise RecoveryConflict(f"Multiple destination reports match session {session_id}.")
    if any(attempt.get("session_id") != session_id for attempt in source_attempts):
        raise RecoveryConflict(f"Source attempts do not consistently match session {session_id}.")

    should_insert_report = not existing_reports
    if existing_reports:
        current = existing_reports[0]
        if current.get("user_id") != target_user_id:
            raise RecoveryConflict(f"Session {session_id} is already owned by another user.")
        if not report_matches(source_report | {"user_id": target_user_id}, current):
            raise RecoveryConflict(f"Existing report for session {session_id} differs from source.")

    source_counts = Counter(attempt_identity(row) for row in source_attempts)
    destination_counts = Counter(attempt_identity(row) for row in existing_attempts)
    if destination_counts - source_counts:
        raise RecoveryConflict(f"Destination has extra or different answers for session {session_id}.")

    remaining = destination_counts.copy()
    missing = []
    for row in source_attempts:
        key = attempt_identity(row)
        if remaining[key]:
            remaining[key] -= 1
        else:
            missing.append(row)
    return should_insert_report, missing
