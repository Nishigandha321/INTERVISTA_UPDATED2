"""Dataset loading and deterministic output evaluation for Coding Round."""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "coding_questions.json"
QUESTION_FIELDS = (
    "id", "title", "description", "difficulty", "topic", "constraints", "function_name",
    "input_format", "output_format", "starter_code_cpp", "starter_code_python",
)
PUBLIC_TEST_CASES = 3
TOTAL_TEST_CASES = 5


class CodingDatasetError(RuntimeError):
    pass


def load_question_bank() -> list[dict[str, Any]]:
    try:
        data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.exception("Unable to load Coding Round question bank")
        raise CodingDatasetError("The coding question bank is temporarily unavailable.") from exc
    if not isinstance(data, list) or len(data) < 2:
        raise CodingDatasetError("The coding question bank is invalid.")

    seen_ids: set[int] = set()
    for question in data:
        if not isinstance(question, dict):
            raise CodingDatasetError("The coding question bank is invalid.")
        question_id = question.get("id")
        if not isinstance(question_id, int) or question_id in seen_ids:
            raise CodingDatasetError("The coding question bank is invalid.")
        seen_ids.add(question_id)
        if any(not isinstance(question.get(field), str) for field in QUESTION_FIELDS if field != "id"):
            raise CodingDatasetError("The coding question bank is invalid.")
        if not isinstance(question.get("reference_solution_cpp"), str) or not isinstance(
            question.get("reference_solution_python"), str
        ):
            raise CodingDatasetError("The coding question bank is invalid.")
        cases = question.get("test_cases")
        if not isinstance(cases, list) or len(cases) != TOTAL_TEST_CASES:
            raise CodingDatasetError(f"Coding question {question_id} must contain exactly {TOTAL_TEST_CASES} test cases.")
        if sum(1 for case in cases if isinstance(case, dict) and case.get("is_sample") is True) != PUBLIC_TEST_CASES:
            raise CodingDatasetError(f"Coding question {question_id} must contain exactly {PUBLIC_TEST_CASES} public test cases.")
        if sum(1 for case in cases if isinstance(case, dict) and case.get("is_sample") is False) != TOTAL_TEST_CASES - PUBLIC_TEST_CASES:
            raise CodingDatasetError(f"Coding question {question_id} must contain exactly {TOTAL_TEST_CASES - PUBLIC_TEST_CASES} hidden test cases.")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(
                case.get("expected_output"), str
            ):
                raise CodingDatasetError("The coding question bank contains an invalid test case.")
    return data


def find_question(question_id: int) -> dict[str, Any] | None:
    return next((q for q in load_question_bank() if q["id"] == question_id), None)


def select_question_ids(excluded_question_ids: set[int] | None = None) -> list[int]:
    question_ids = [question["id"] for question in load_question_bank()]
    excluded = excluded_question_ids or set()
    eligible = [question_id for question_id in question_ids if question_id not in excluded]
    if len(eligible) < 2:
        eligible = question_ids
    return secrets.SystemRandom().sample(eligible, 2)


def public_question(question: dict[str, Any]) -> dict[str, Any]:
    """Allowlist the prompt and its first public sample while an attempt is active."""
    public = {field: question[field] for field in QUESTION_FIELDS}
    cases = question.get("test_cases") or []
    public_cases = [case for case in cases if case.get("is_sample") is True][:PUBLIC_TEST_CASES]
    if public_cases:
        public["examples"] = [{"input": case["input"], "output": case["expected_output"]} for case in public_cases]
    return public


def normalize_output(value: str | None) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def evaluate_execution(
    test_number: int, execution: dict[str, Any], expected_output: str
) -> dict[str, Any]:
    status = execution.get("status") or {}
    status_id = execution.get("status_id", status.get("id"))
    description = str(status.get("description") or "Execution Error")
    stdout = str(execution.get("stdout") or "")
    stderr = str(execution.get("stderr") or "")
    compile_output = str(execution.get("compile_output") or "")
    compiler_warning = ""

    if status_id == 3 or description.lower() == "accepted":
        passed = normalize_output(stdout) == normalize_output(expected_output)
        verdict = "Accepted" if passed else "Wrong Answer"
        # OnlineCompiler may return compiler warnings in stderr despite a successful
        # compile and execution. Keep them separate from a wrong-answer explanation.
        compiler_warning = stderr
        stderr = ""
    else:
        passed = False
        description_lower = description.lower()
        if "compil" in description_lower:
            verdict = "Compilation Error"
        elif "time limit" in description_lower or "wall time" in description_lower:
            verdict = "Time Limit Exceeded"
        elif "runtime" in description_lower or "non-zero" in description_lower:
            verdict = "Runtime Error"
        elif "wrong answer" in description_lower:
            verdict = "Wrong Answer"
        else:
            verdict = description[:80] or "Execution Error"

    result = {
        "test_number": test_number,
        "passed": passed,
        "status": verdict,
        "stdout": stdout[:4000],
        "stderr": (compile_output or stderr)[:4000],
        "compiler_warning": compiler_warning[:4000],
        "execution_time": execution.get("time"),
        "memory": execution.get("memory"),
        "retryable": False,
    }
    if verdict == "Wrong Answer":
        result["expected_output"] = str(expected_output)[:4000]
    return result


def evaluate_service_error(test_number: int, message: str) -> dict[str, Any]:
    return {
        "test_number": test_number,
        "passed": False,
        "status": "Execution Provider Error",
        "stdout": "",
        "stderr": message[:400],
        "execution_time": None,
        "memory": None,
        "retryable": True,
    }


def score_tests(results: list[dict[str, Any]]) -> tuple[int, float]:
    passed = sum(1 for result in results if result.get("passed"))
    total = len(results) or TOTAL_TEST_CASES
    return passed, round(passed * 100.0 / total, 2)
