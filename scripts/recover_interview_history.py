"""Safely recover the verified candidate's SQLite interview history to Supabase."""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.orm import Session

from database import engine
from models import Interview, InterviewAttempt, User, UserProfile
from services.interview_history_recovery import (
    RecoveryConflict,
    attempt_identity,
    plan_session,
    report_matches,
)


SOURCE_USER_ID = 8
TARGET_USER_ID = 12
TARGET_USERNAME = "demo_data_analyst_2026"
EXPECTED_SESSION_COUNT = 7


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parse_json_object(raw: str, label: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RecoveryConflict(f"Invalid JSON in {label}.") from exc
    if not isinstance(value, dict):
        raise RecoveryConflict(f"Expected JSON object in {label}.")
    return value


def _identity_matches(source_conn: sqlite3.Connection, db: Session) -> None:
    local_user_rows = source_conn.execute(
        "SELECT id, username FROM users WHERE username = ?", (TARGET_USERNAME,)
    ).fetchall()
    remote_users = db.query(User).filter(User.username == TARGET_USERNAME).all()
    if len(local_user_rows) != 1 or local_user_rows[0]["id"] != SOURCE_USER_ID:
        raise RecoveryConflict("Source username does not uniquely resolve to local user ID 8.")
    if len(remote_users) != 1 or remote_users[0].id != TARGET_USER_ID:
        raise RecoveryConflict("Destination username does not uniquely resolve to PostgreSQL user ID 12.")

    local_profile = source_conn.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?", (SOURCE_USER_ID,)
    ).fetchone()
    remote_profile = db.query(UserProfile).filter(UserProfile.user_id == TARGET_USER_ID).one_or_none()
    if local_profile is None or remote_profile is None:
        raise RecoveryConflict("A required source or destination profile is missing.")

    email = (local_profile["email"] or "").strip().casefold()
    remote_email = (remote_profile.email or "").strip().casefold()
    if not email or email != remote_email:
        raise RecoveryConflict("The nonempty profile email does not match across stores.")

    local_email_count = source_conn.execute(
        "SELECT count(*) FROM user_profiles WHERE lower(trim(email)) = ?", (email,)
    ).fetchone()[0]
    remote_email_count = db.query(UserProfile).filter(
        UserProfile.email.isnot(None),
    ).all()
    remote_email_count = sum(
        1 for profile in remote_email_count
        if (profile.email or "").strip().casefold() == email
    )
    if local_email_count != 1 or remote_email_count != 1:
        raise RecoveryConflict("The matching email is not unique in both stores.")

    for field in ("role_applied_for", "current_designation", "extracted_skills"):
        left = local_profile[field]
        right = getattr(remote_profile, field)
        if (left or "").strip() != (right or "").strip():
            raise RecoveryConflict(f"Core profile field {field} differs across stores.")


def _load_source(source_path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    if not source_path.is_file():
        raise RecoveryConflict("The SQLite source database is unavailable.")
    conn = sqlite3.connect(f"file:{source_path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        reports = []
        for row in conn.execute(
            "SELECT id,user_id,role,date,score,report_json FROM interviews WHERE user_id = ? ORDER BY date,id",
            (SOURCE_USER_ID,),
        ):
            report_data = _parse_json_object(row["report_json"], f"interviews.id={row['id']}")
            session_id = report_data.get("session_id")
            if not session_id:
                raise RecoveryConflict(f"Source report {row['id']} has no session ID.")
            detail = report_data.get("detailed_answers")
            if not isinstance(detail, list):
                raise RecoveryConflict(f"Source report {row['id']} has no detailed-answer list.")
            reports.append({
                "id": row["id"], "user_id": row["user_id"], "role": row["role"],
                "date": datetime.fromisoformat(row["date"]) if row["date"] else None,
                "score": row["score"], "report_json": row["report_json"],
                "session_id": session_id, "detailed_answers_count": len(detail),
            })
        if len(reports) != EXPECTED_SESSION_COUNT:
            raise RecoveryConflict(
                f"Expected {EXPECTED_SESSION_COUNT} source reports, found {len(reports)}."
            )
        if len({report["session_id"] for report in reports}) != len(reports):
            raise RecoveryConflict("Source reports contain duplicate session IDs.")

        report_sessions = {report["session_id"] for report in reports}
        attempts_by_session = {session_id: [] for session_id in report_sessions}
        all_source_attempts = []
        for row in conn.execute(
            "SELECT id,user_id,role,topic,difficulty,answer,feedback,timestamp "
            "FROM interview_attempts WHERE user_id = ? ORDER BY timestamp,id",
            (SOURCE_USER_ID,),
        ):
            attempt = dict(row)
            feedback = _parse_json_object(attempt["feedback"], f"interview_attempts.id={attempt['id']}")
            session_id = feedback.get("session_id")
            if session_id not in attempts_by_session:
                raise RecoveryConflict("A source answer attempt is not linked to one of the seven reports.")
            attempt["session_id"] = session_id
            attempt["timestamp"] = datetime.fromisoformat(attempt["timestamp"]) if attempt["timestamp"] else None
            attempts_by_session[session_id].append(attempt)
            all_source_attempts.append(attempt)

        reports_by_session = {report["session_id"]: report for report in reports}
        for session_id, attempts in attempts_by_session.items():
            if not attempts:
                raise RecoveryConflict(f"Source report session {session_id} has no answer attempts.")
            if reports_by_session[session_id]["detailed_answers_count"] != len(attempts):
                raise RecoveryConflict(f"Report/answer counts disagree for source session {session_id}.")
        return reports, attempts_by_session
    finally:
        conn.close()


def _destination_session_rows(db: Session, session_id: str) -> tuple[list[dict], list[dict]]:
    report_rows = []
    for row in db.query(Interview).filter(Interview.report_json.contains(session_id)).all():
        report = _parse_json_object(row.report_json, f"destination interviews.id={row.id}")
        if report.get("session_id") == session_id:
            report_rows.append({
                "id": row.id, "user_id": row.user_id, "role": row.role,
                "date": row.date, "score": row.score, "report_json": row.report_json,
                "session_id": session_id,
            })

    attempt_rows = []
    for row in db.query(InterviewAttempt).filter(InterviewAttempt.feedback.contains(session_id)).all():
        feedback = _parse_json_object(row.feedback, f"destination interview_attempts.id={row.id}")
        if feedback.get("session_id") == session_id:
            attempt_rows.append({
                "id": row.id, "user_id": row.user_id, "role": row.role,
                "topic": row.topic, "difficulty": row.difficulty, "answer": row.answer,
                "feedback": row.feedback, "timestamp": row.timestamp, "session_id": session_id,
            })
    return report_rows, attempt_rows


def _backup_path() -> Path:
    # Use the isolated per-session temp root so the backup path is writable in
    # managed workspaces and remains outside the repository.
    folder = Path(tempfile.gettempdir()) / "INTERVISTA" / "migration_backups"
    folder.mkdir(parents=True, exist_ok=True)
    backup = folder / f"interview_history_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    backup.mkdir()
    return backup


def _write_private_json(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, default=_json_default, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())


def recover(source_path: Path, apply: bool, confirm_user_map: str | None = None) -> dict:
    if engine.dialect.name != "postgresql":
        raise RecoveryConflict("Configured destination is not PostgreSQL; refusing migration.")
    if apply and confirm_user_map != "8:12":
        raise RecoveryConflict("Applying requires explicit confirmation of user mapping 8:12.")

    reports, attempts_by_session = _load_source(source_path)
    plans = []
    backup_folder = None
    with Session(bind=engine, expire_on_commit=False) as db:
        db.connection().exec_driver_sql("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        db.connection().exec_driver_sql(
            "SELECT pg_advisory_xact_lock(hashtext('intervista-history-recovery-8-to-12'))"
        )
        source_conn = sqlite3.connect(f"file:{source_path.resolve().as_posix()}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            _identity_matches(source_conn, db)
        finally:
            source_conn.close()

        target = db.query(User).filter(User.id == TARGET_USER_ID).one_or_none()
        if target is None or target.username != TARGET_USERNAME:
            raise RecoveryConflict("Destination user ID 12 no longer matches the verified username.")

        for source_report in reports:
            sid = source_report["session_id"]
            existing_reports, existing_attempts = _destination_session_rows(db, sid)
            should_report, missing_attempts = plan_session(
                source_report,
                attempts_by_session[sid],
                existing_reports,
                existing_attempts,
                TARGET_USER_ID,
            )
            plans.append((source_report, should_report, missing_attempts))

        if not apply:
            return {
                "status": "dry_run_ok", "source_reports": len(reports),
                "source_attempts": sum(map(len, attempts_by_session.values())),
                "reports_to_insert": sum(should for _, should, _ in plans),
                "attempts_to_insert": sum(len(missing) for _, _, missing in plans),
                "existing_target_reports": db.query(Interview).filter(Interview.user_id == TARGET_USER_ID).count(),
                "backup_created": False,
            }

        backup_folder = _backup_path()
        destination_reports = [
            {"id": r.id, "user_id": r.user_id, "role": r.role, "date": r.date,
             "score": r.score, "report_json": r.report_json}
            for r in db.query(Interview).filter(Interview.user_id == TARGET_USER_ID).all()
        ]
        destination_attempts = [
            {"id": a.id, "user_id": a.user_id, "role": a.role, "topic": a.topic,
             "difficulty": a.difficulty, "answer": a.answer, "feedback": a.feedback,
             "timestamp": a.timestamp}
            for a in db.query(InterviewAttempt).filter(InterviewAttempt.user_id == TARGET_USER_ID).all()
        ]
        source_attempts = [a for session_attempts in attempts_by_session.values() for a in session_attempts]
        _write_private_json(backup_folder / "source_history.json", {
            "source_user_id": SOURCE_USER_ID,
            "reports": [{k: r[k] for k in ("id", "user_id", "role", "date", "score", "report_json")} for r in reports],
            "attempts": [{k: a[k] for k in ("id", "user_id", "role", "topic", "difficulty", "answer", "feedback", "timestamp")} for a in source_attempts],
        })
        _write_private_json(backup_folder / "destination_history_before.json", {
            "target_user_id": TARGET_USER_ID,
            "reports": destination_reports,
            "attempts": destination_attempts,
        })

        imported_reports = []
        imported_attempt_count = 0
        for source_report, should_report, missing_attempts in plans:
            if should_report:
                row = Interview(
                    user_id=TARGET_USER_ID, role=source_report["role"],
                    date=source_report["date"], score=source_report["score"],
                    report_json=source_report["report_json"],
                )
                db.add(row)
                db.flush()
                imported_reports.append({"source_report_id": source_report["id"], "destination_report_id": row.id})
            for source_attempt in missing_attempts:
                db.add(InterviewAttempt(
                    user_id=TARGET_USER_ID, role=source_attempt["role"],
                    topic=source_attempt["topic"], difficulty=source_attempt["difficulty"],
                    answer=source_attempt["answer"], feedback=source_attempt["feedback"],
                    timestamp=source_attempt["timestamp"],
                ))
                imported_attempt_count += 1

        db.flush()

        # Verify exact session payloads and ownership while the transaction is open.
        for source_report, _, _ in plans:
            sid = source_report["session_id"]
            dest_reports, dest_attempts = _destination_session_rows(db, sid)
            if len(dest_reports) != 1 or not report_matches(source_report | {"user_id": TARGET_USER_ID}, dest_reports[0]):
                raise RecoveryConflict(f"Post-insert report verification failed for session {sid}.")
            if Counter(attempt_identity(a) for a in dest_attempts) != Counter(
                attempt_identity(a) for a in attempts_by_session[sid]
            ):
                raise RecoveryConflict(f"Post-insert answer verification failed for session {sid}.")

        history_count = db.query(Interview).filter(Interview.user_id == TARGET_USER_ID).count()
        db.commit()
        return {
            "status": "applied", "reports_imported": len(imported_reports),
            "attempts_imported": imported_attempt_count,
            "destination_report_ids": imported_reports,
            "target_history_report_count": history_count,
            "backup_folder": str(backup_folder), "backup_created": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and apply the verified migration.")
    parser.add_argument("--confirm-user-map", help="Must be exactly 8:12 when --apply is specified.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "jobprepmate.db")
    args = parser.parse_args()
    try:
        result = recover(args.source, args.apply, args.confirm_user_map)
    except RecoveryConflict as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
