"""Isolated, authenticated Coding Round routes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_
from sqlalchemy.orm import Session

from models import CodingAttempt, CodingQuestionResult
from services.coding.evaluator import (
    CodingDatasetError,
    evaluate_execution,
    evaluate_service_error,
    find_question,
    load_question_bank,
    public_question,
    score_tests,
    select_question_ids,
    PUBLIC_TEST_CASES,
    TOTAL_TEST_CASES,
)
from services.coding.execution_service import (
    CodingExecutionConfigurationError,
    CodingExecutionError,
    coding_execution_service,
)
from services.coding.harnesses import build_execution_source, has_entrypoint

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
MAX_SOURCE_BYTES = 32 * 1024
MAX_INPUT_BYTES = 16 * 1024
MAX_SUBMISSIONS_PER_QUESTION = 2
CODING_LOCKS: dict[int, asyncio.Lock] = {}


def _question_response(question: dict[str, Any], row: CodingQuestionResult) -> dict[str, Any]:
    response = {
        "question": public_question(question),
        "question_order": row.question_order,
        "execution_count": row.execution_count,
        "submission_count": row.submission_count,
        "submission_limit": MAX_SUBMISSIONS_PER_QUESTION,
    }
    if row.source_code is not None and not has_entrypoint(row.language or "cpp", row.source_code):
        response["saved_source_code"] = row.source_code
        response["language"] = row.language or "cpp"
    if isinstance(row.test_results, list):
        response["saved_results"] = [
            {key: value for key, value in result.items() if key != "retryable"}
            for result in row.test_results
            if result.get("test_number", 0) <= PUBLIC_TEST_CASES
        ]
    return response


def _current_question(db: Session, attempt: CodingAttempt) -> CodingQuestionResult | None:
    return (
        db.query(CodingQuestionResult)
        .filter(
            CodingQuestionResult.coding_attempt_id == attempt.id,
            CodingQuestionResult.submitted_at.is_(None),
        )
        .order_by(CodingQuestionResult.question_order.asc())
        .first()
    )


def _discard_legacy_full_program(row: CodingQuestionResult) -> bool:
    """Clear old full-program source/results from rounds created before function mode."""
    if row.source_code is None or not has_entrypoint(row.language or "cpp", row.source_code):
        return False
    row.source_code = None
    row.source_hash = None
    row.counted_source_hash = None
    row.submission_count = 0
    row.test_results = None
    return True


def _question_results(row: CodingQuestionResult) -> list[dict[str, Any]]:
    results = row.test_results
    # Completed reports created under the former ten-test configuration remain
    # readable with their original saved totals and public-result visibility.
    if isinstance(results, list) and row.submitted_at is not None and len(results) > TOTAL_TEST_CASES:
        return results
    if not isinstance(results, list) or len(results) != TOTAL_TEST_CASES:
        return [
            {
                "test_number": number,
                "passed": False,
                "status": "Not Run",
                "stdout": "",
                "stderr": "",
                "execution_time": None,
                "memory": None,
                "retryable": False,
            }
            for number in range(1, TOTAL_TEST_CASES + 1)
        ]
    return results


def _attempt_progress(db: Session, attempt: CodingAttempt) -> dict[str, Any]:
    row = _current_question(db, attempt)
    if row is None:
        return {"attempt_id": attempt.id, "status": attempt.status, "is_complete": attempt.status == "completed"}
    question = find_question(row.question_id)
    if question is None:
        raise HTTPException(status_code=503, detail="The assigned coding question is unavailable.")
    if _discard_legacy_full_program(row):
        db.add(row)
        db.commit()
    return {
        "attempt_id": attempt.id,
        "status": attempt.status,
        "is_complete": False,
        "attempt_execution_count": attempt.execution_count,
        "submission_count": row.submission_count,
        "submission_limit": MAX_SUBMISSIONS_PER_QUESTION,
        **_question_response(question, row),
    }


def _result_payload(db: Session, attempt: CodingAttempt) -> dict[str, Any]:
    rows = (
        db.query(CodingQuestionResult)
        .filter(CodingQuestionResult.coding_attempt_id == attempt.id)
        .order_by(CodingQuestionResult.question_order.asc())
        .all()
    )
    question_results = []
    for row in rows:
        question = find_question(row.question_id)
        if question is None:
            raise HTTPException(status_code=503, detail="A coding question in this report is unavailable.")
        question_results.append({
            "question_order": row.question_order,
            "question_id": row.question_id,
            "title": question["title"],
            "question": question["description"],
            "difficulty": question["difficulty"],
            "topic": question["topic"],
            "language": row.language,
            "passed_test_cases": row.passed_test_cases,
            "total_test_cases": row.total_test_cases,
            "score": row.score,
            "execution_status": row.execution_status,
            "execution_count": row.execution_count,
            "tests": [
                {key: value for key, value in result.items() if key != "retryable"}
                for result in _question_results(row)
                if result.get("test_number", 0) <= PUBLIC_TEST_CASES
            ],
            # Keep the existing report key for API compatibility, but ensure it
            # contains the candidate's final submitted code, never the canonical
            # answer stored in the private question bank.
            "reference_solution": row.source_code or "",
            "submitted_solution": row.source_code or "",
        })
    return {
        "attempt_id": attempt.id,
        "created_at": attempt.created_at.isoformat() if attempt.created_at else None,
        "completed_at": attempt.completed_at.isoformat() if attempt.completed_at else None,
        "status": attempt.status,
        "total_questions": attempt.total_questions,
        "attempted_questions": attempt.attempted_questions,
        "solved_questions": attempt.solved_questions,
        "total_test_cases": attempt.total_test_cases,
        "passed_test_cases": attempt.passed_test_cases,
        "execution_count": attempt.execution_count,
        "overall_score": attempt.overall_score,
        "questions": question_results,
    }


def _validate_submission(body: Any) -> tuple[int, int, str, str]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid coding submission.")
    try:
        attempt_id = int(body.get("attempt_id"))
        question_id = int(body.get("question_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="A valid attempt and question are required.")
    language = body.get("language")
    source_code = body.get("source_code")
    if language not in {"cpp", "python"}:
        raise HTTPException(status_code=400, detail="Choose C++ or Python.")
    if not isinstance(source_code, str) or not source_code.strip():
        raise HTTPException(status_code=400, detail="Enter code before running or submitting.")
    if len(source_code.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Source code exceeds the 32 KB limit.")
    return attempt_id, question_id, language, source_code


def _reserve_execution(
    db: Session,
    attempt_id: int,
    question_row_id: int,
    user_id: int,
    source_hash: str,
    source_code: str,
    language: str,
) -> None:
    """Record one provider request; this counter does not consume candidate submissions."""
    attempt_count = (
        db.query(CodingAttempt)
        .filter(
            CodingAttempt.id == attempt_id,
            CodingAttempt.user_id == user_id,
            CodingAttempt.status == "in_progress",
        )
        .update({CodingAttempt.execution_count: CodingAttempt.execution_count + 1}, synchronize_session=False)
    )
    row_count = (
        db.query(CodingQuestionResult)
        .filter(
            CodingQuestionResult.id == question_row_id,
            CodingQuestionResult.coding_attempt_id == attempt_id,
            CodingQuestionResult.submitted_at.is_(None),
            or_(CodingQuestionResult.source_hash.is_(None), CodingQuestionResult.source_hash == source_hash),
        )
        .update({
            CodingQuestionResult.execution_count: CodingQuestionResult.execution_count + 1,
            CodingQuestionResult.source_hash: source_hash,
            CodingQuestionResult.source_code: source_code,
            CodingQuestionResult.language: language,
        }, synchronize_session=False)
    )
    if attempt_count != 1 or row_count != 1:
        db.rollback()
        raise HTTPException(
            status_code=429,
            detail="Execution limit reached for this question or Coding Round. No additional execution was started.",
        )
    db.commit()


async def _run_or_submit(
    *, request: Request,
    db: Session,
    user: Any,
    finalize: bool,
) -> JSONResponse:
    attempt_id, question_id, language, source_code = _validate_submission(await request.json())
    lock = CODING_LOCKS.setdefault(attempt_id, asyncio.Lock())
    async with lock:
        attempt = (
            db.query(CodingAttempt)
            .filter(CodingAttempt.id == attempt_id, CodingAttempt.user_id == user.id)
            .first()
        )
        if not attempt or attempt.status != "in_progress":
            raise HTTPException(status_code=404, detail="Active Coding Round not found.")
        row = _current_question(db, attempt)
        if row is None or row.question_id != question_id:
            raise HTTPException(status_code=409, detail="This is not the current question in this Coding Round.")
        question = find_question(row.question_id)
        if not question:
            raise HTTPException(status_code=503, detail="The assigned coding question is unavailable.")
        if _discard_legacy_full_program(row):
            db.add(row)
            db.commit()
        try:
            execution_source = build_execution_source(question, language, source_code)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if any(len(case["input"].encode("utf-8")) > MAX_INPUT_BYTES for case in question["test_cases"]):
            raise HTTPException(status_code=503, detail="The coding question contains an oversized test case.")

        source_hash = hashlib.sha256(f"{language}\0{source_code}".encode("utf-8")).hexdigest()
        if row.source_hash and row.source_hash != source_hash:
            if any(result.get("retryable") or result.get("status") == "Not Run" for result in _question_results(row)):
                raise HTTPException(status_code=409, detail="Retry the same code until the execution service completes all tests.")
            if row.submission_count >= MAX_SUBMISSIONS_PER_QUESTION:
                raise HTTPException(status_code=429, detail="This question has used both submissions.")
            row.test_results = None
            row.source_hash = source_hash
            row.source_code = source_code
            row.language = language
            db.add(row)
            db.commit()
        elif not row.source_hash:
            row.source_hash = source_hash
            row.source_code = source_code
            row.language = language
            db.add(row)
            db.commit()

        results = _question_results(row)
        completed_cached = all(not result.get("retryable") and result.get("status") != "Not Run" for result in results)
        if not completed_cached:
            try:
                coding_execution_service.validate_language(language)
            except (CodingExecutionConfigurationError, CodingExecutionError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            for index, case in enumerate(question["test_cases"]):
                current = results[index]
                if not current.get("retryable") and current.get("status") != "Not Run":
                    continue
                _reserve_execution(db, attempt.id, row.id, user.id, source_hash, source_code, language)
                db.refresh(attempt)
                db.refresh(row)
                try:
                    execution = await coding_execution_service.execute(execution_source, language, case["input"])
                    results[index] = evaluate_execution(index + 1, execution, case["expected_output"])
                except CodingExecutionError as exc:
                    logger.warning("Coding provider execution failed: attempt_id=%s question_id=%s test=%s error=%s", attempt.id, question_id, index + 1, str(exc))
                    results[index] = evaluate_service_error(index + 1, str(exc))
                    row.test_results = list(results)
                    db.add(row)
                    db.commit()
                    break
                row.test_results = list(results)
                db.add(row)
                db.commit()
            row.test_results = list(results)
            row.language = language
            db.add(row)
            db.commit()
            db.refresh(row)
            db.refresh(attempt)
            results = _question_results(row)

        has_retryable = any(result.get("retryable") or result.get("status") == "Not Run" for result in results)
        public_results = [
            {key: value for key, value in result.items() if key != "retryable"}
            for result in results if result.get("test_number", 0) <= PUBLIC_TEST_CASES
        ]
        if has_retryable:
            return JSONResponse(status_code=503, content={
                "detail": "The execution service did not complete all tests. Retry the same code; this does not use a submission.",
                "incomplete": True,
                "results": public_results,
                "execution_count": row.execution_count,
                "submission_count": row.submission_count,
                "submission_limit": MAX_SUBMISSIONS_PER_QUESTION,
            })

        if row.counted_source_hash != source_hash:
            if row.submission_count >= MAX_SUBMISSIONS_PER_QUESTION:
                raise HTTPException(status_code=429, detail="This question has used both submissions.")
            row.submission_count += 1
            row.counted_source_hash = source_hash
            db.add(row)
            db.commit()
            db.refresh(row)

        passed, score = score_tests(results)
        response: dict[str, Any] = {
            "attempt_id": attempt.id,
            "question_id": row.question_id,
            "question_order": row.question_order,
            "language": language,
            "execution_count": row.execution_count,
            "attempt_execution_count": attempt.execution_count,
            "submission_count": row.submission_count,
            "submission_limit": MAX_SUBMISSIONS_PER_QUESTION,
            "results": public_results,
            "passed_test_cases": passed,
            "total_test_cases": TOTAL_TEST_CASES,
            "score": score,
        }
        if not finalize:
            return JSONResponse(content=response)

        if passed < TOTAL_TEST_CASES and row.submission_count < MAX_SUBMISSIONS_PER_QUESTION:
            response["can_retry"] = True
            response["message"] = "Some tests failed. Revise your code and use your second submission."
            return JSONResponse(content=response)

        row.language = language
        row.passed_test_cases = passed
        row.total_test_cases = TOTAL_TEST_CASES
        row.score = score
        row.execution_status = "Accepted" if passed == TOTAL_TEST_CASES else "Partial" if passed else "Not Solved"
        row.submitted_at = datetime.utcnow()
        row.test_results = list(results)
        attempt.attempted_questions += 1
        attempt.passed_test_cases += passed
        if passed == TOTAL_TEST_CASES:
            attempt.solved_questions += 1
        next_row = (
            db.query(CodingQuestionResult)
            .filter(
                CodingQuestionResult.coding_attempt_id == attempt.id,
                CodingQuestionResult.question_order > row.question_order,
            )
            .order_by(CodingQuestionResult.question_order.asc())
            .first()
        )
        if next_row is None:
            attempt.status = "completed"
            attempt.completed_at = datetime.utcnow()
            attempt.overall_score = round(attempt.passed_test_cases * 100.0 / (2 * TOTAL_TEST_CASES), 2)
            db.add(row)
            db.add(attempt)
            db.commit()
            response["completed"] = True
            response["result_url"] = f"/coding/attempt/{attempt.id}/result"
            return JSONResponse(content=response)

        db.add(row)
        db.add(attempt)
        db.commit()
        next_question = find_question(next_row.question_id)
        if next_question is None:
            raise HTTPException(status_code=503, detail="The next coding question is unavailable.")
        if _discard_legacy_full_program(next_row):
            db.add(next_row)
            db.commit()
        response["completed"] = False
        response["next"] = {
            **_question_response(next_question, next_row),
            "attempt_execution_count": attempt.execution_count,
        }
        return JSONResponse(content=response)


def create_coding_router(
    get_current_user: Callable[..., Any],
    get_db: Callable[..., Any],
) -> APIRouter:
    """Build routes using the app's existing cookie authentication and DB dependency."""
    router = APIRouter()

    def require_user(request: Request, db: Session = Depends(get_db)):
        user = get_current_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Not logged in")
        return user

    @router.get("/coding", response_class=HTMLResponse)
    def coding_page(request: Request, user=Depends(require_user)):
        return templates.TemplateResponse(request, "coding_round.html", {"request": request, "username": user.username})

    @router.get("/api/coding/questions")
    def coding_questions(user=Depends(require_user)):
        try:
            questions = load_question_bank()
        except CodingDatasetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return [{"id": q["id"], "title": q["title"], "difficulty": q["difficulty"], "topic": q["topic"]} for q in questions]

    @router.get("/api/coding/questions/{question_id}")
    def coding_question(question_id: int, user=Depends(require_user)):
        try:
            question = find_question(question_id)
        except CodingDatasetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if question is None:
            raise HTTPException(status_code=404, detail="Coding question not found.")
        return public_question(question)

    @router.post("/api/coding/attempt/start")
    def start_attempt(db: Session = Depends(get_db), user=Depends(require_user)):
        active = (
            db.query(CodingAttempt)
            .filter(CodingAttempt.user_id == user.id, CodingAttempt.status == "in_progress")
            .order_by(CodingAttempt.created_at.desc())
            .first()
        )
        if active:
            rows = db.query(CodingQuestionResult).filter_by(coding_attempt_id=active.id).all()
            if len(rows) != 2:
                raise HTTPException(status_code=503, detail="The active Coding Round is incomplete. Please contact support.")
            return _attempt_progress(db, active)

        try:
            previous_attempt = (
                db.query(CodingAttempt)
                .filter(CodingAttempt.user_id == user.id, CodingAttempt.status == "completed")
                .order_by(CodingAttempt.created_at.desc(), CodingAttempt.id.desc())
                .first()
            )
            previous_question_ids: set[int] = set()
            if previous_attempt:
                previous_question_ids = {
                    row.question_id
                    for row in db.query(CodingQuestionResult.question_id)
                    .filter(CodingQuestionResult.coding_attempt_id == previous_attempt.id)
                    .all()
                }
            question_ids = select_question_ids(previous_question_ids)
        except CodingDatasetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        attempt = CodingAttempt(user_id=user.id, total_questions=2, total_test_cases=2 * TOTAL_TEST_CASES, status="in_progress")
        db.add(attempt)
        db.flush()
        for order, question_id in enumerate(question_ids, start=1):
            db.add(CodingQuestionResult(
                coding_attempt_id=attempt.id,
                question_id=question_id,
                question_order=order,
                total_test_cases=TOTAL_TEST_CASES,
                execution_status="pending",
                test_results=None,
            ))
        db.commit()
        db.refresh(attempt)
        return _attempt_progress(db, attempt)

    @router.post("/api/coding/run")
    async def run_code(request: Request, db: Session = Depends(get_db), user=Depends(require_user)):
        return await _run_or_submit(request=request, db=db, user=user, finalize=False)

    @router.post("/api/coding/submit")
    async def submit_code(request: Request, db: Session = Depends(get_db), user=Depends(require_user)):
        return await _run_or_submit(request=request, db=db, user=user, finalize=True)

    @router.get("/api/coding/attempt/{attempt_id}/result")
    def coding_result_json(attempt_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
        attempt = db.query(CodingAttempt).filter_by(id=attempt_id, user_id=user.id).first()
        if not attempt or attempt.status != "completed":
            raise HTTPException(status_code=404, detail="Completed Coding Round report not found.")
        return _result_payload(db, attempt)

    @router.get("/coding/attempt/{attempt_id}/result", response_class=HTMLResponse)
    def coding_result_page(request: Request, attempt_id: int, db: Session = Depends(get_db), user=Depends(require_user)):
        attempt = db.query(CodingAttempt).filter_by(id=attempt_id, user_id=user.id).first()
        if not attempt or attempt.status != "completed":
            raise HTTPException(status_code=404, detail="Completed Coding Round report not found.")
        report = _result_payload(db, attempt)
        return templates.TemplateResponse(request, "coding_result.html", {
            "request": request, "username": user.username, "report": report,
        })

    return router
