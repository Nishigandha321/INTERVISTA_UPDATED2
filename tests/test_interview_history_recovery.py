"""Non-destructive session reconciliation tests for history recovery."""

from datetime import datetime

import pytest

from services.interview_history_recovery import RecoveryConflict, plan_session


def _report():
    return {
        "id": 27, "user_id": 12, "role": "Junior Data Analyst",
        "date": datetime(2026, 10, 9, 21, 22), "score": 61.8,
        "report_json": '{"session_id":"sid-1"}', "session_id": "sid-1",
    }


def _attempt(answer="answer one"):
    return {
        "user_id": 12, "role": "Junior Data Analyst", "topic": "voice-interview",
        "difficulty": "adaptive", "answer": answer,
        "feedback": '{"session_id":"sid-1","question":"Q1"}',
        "timestamp": datetime(2026, 10, 9, 21, 20), "session_id": "sid-1",
    }


def test_new_session_plans_report_and_all_answer_rows_without_mutating_inputs():
    report = _report()
    attempts = [_attempt(), _attempt("answer two")]
    should_insert_report, missing = plan_session(report, attempts, [], [], target_user_id=12)

    assert should_insert_report is True
    assert missing == attempts
    assert report["id"] == 27


def test_exact_existing_session_is_idempotent():
    report = _report()
    existing_report = dict(report)
    existing_attempts = [_attempt(), _attempt("answer two")]

    should_insert_report, missing = plan_session(
        report, existing_attempts, [existing_report], existing_attempts, target_user_id=12
    )

    assert should_insert_report is False
    assert missing == []


def test_partial_exact_answer_copy_plans_only_missing_answer():
    source_attempts = [_attempt(), _attempt("answer two")]

    should_insert_report, missing = plan_session(
        _report(), source_attempts, [], [source_attempts[0]], target_user_id=12
    )

    assert should_insert_report is True
    assert missing == [source_attempts[1]]


def test_other_owner_or_different_existing_payload_stops_without_overwrite():
    report = _report()
    other_owner = dict(report, user_id=2)
    with pytest.raises(RecoveryConflict, match="another user"):
        plan_session(report, [_attempt()], [other_owner], [_attempt()], target_user_id=12)

    altered = dict(report, score=0.0)
    with pytest.raises(RecoveryConflict, match="differs from source"):
        plan_session(report, [_attempt()], [altered], [_attempt()], target_user_id=12)


def test_extra_or_different_destination_answer_stops():
    with pytest.raises(RecoveryConflict, match="extra or different answers"):
        plan_session(
            _report(), [_attempt()], [], [_attempt("not the source answer")], target_user_id=12
        )
