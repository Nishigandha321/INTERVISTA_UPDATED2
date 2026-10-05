import asyncio
import json
from pathlib import Path
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from database import Base
from models import CodingAttempt, CodingQuestionResult, User
from services.coding import evaluator, routes
from services.coding.evaluator import (
    CodingDatasetError,
    evaluate_execution,
    find_question,
    load_question_bank,
    normalize_output,
    score_tests,
)
from services.coding.execution_service import (
    CodingExecutionConfigurationError,
    CodingExecutionError,
    CodingExecutionService,
)


@pytest.fixture
def client_and_sessions(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with testing_sessions() as db:
        db.add(User(username="coding-user", password="not-used"))
        db.commit()

    def override_get_db():
        db = testing_sessions()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(routes.create_coding_router(main.get_current_user, main.get_db))
    app.dependency_overrides[main.get_db] = override_get_db
    routes.CODING_LOCKS.clear()
    with TestClient(app) as client:
        client.cookies.set("user", "coding-user")
        yield client, testing_sessions, app
    engine.dispose()


def test_schema_upgrade_adds_only_source_code_and_preserves_existing_result_data():
    legacy_engine = create_engine("sqlite://")
    try:
        with legacy_engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE coding_question_results (
                    id INTEGER PRIMARY KEY,
                    coding_attempt_id INTEGER NOT NULL,
                    question_id INTEGER NOT NULL,
                    question_order INTEGER NOT NULL,
                    language VARCHAR,
                    passed_test_cases INTEGER NOT NULL DEFAULT 0,
                    total_test_cases INTEGER NOT NULL DEFAULT 3,
                    score FLOAT NOT NULL DEFAULT 0,
                    execution_status VARCHAR NOT NULL DEFAULT 'pending',
                    execution_count INTEGER NOT NULL DEFAULT 0,
                    source_hash VARCHAR(64),
                    test_results JSON,
                    submitted_at DATETIME
                )
            """))
            conn.execute(text("""
                INSERT INTO coding_question_results
                    (id, coding_attempt_id, question_id, question_order, execution_count, source_hash)
                VALUES (1, 7, 12, 1, 2, 'existing-hash')
            """))

        before = {column["name"] for column in inspect(legacy_engine).get_columns("coding_question_results")}
        expected = {column.name for column in CodingQuestionResult.__table__.columns}
        assert expected - before == {"source_code"}
        assert before - expected == set()

        main.ensure_database_schema(legacy_engine)

        after = {column["name"] for column in inspect(legacy_engine).get_columns("coding_question_results")}
        assert after == expected
        with legacy_engine.connect() as conn:
            row = conn.execute(text("""
                SELECT id, coding_attempt_id, question_id, question_order, execution_count, source_hash, source_code
                FROM coding_question_results WHERE id = 1
            """)).one()
        assert tuple(row) == (1, 7, 12, 1, 2, "existing-hash", None)
    finally:
        legacy_engine.dispose()


@pytest.fixture
def successful_onlinecompiler(monkeypatch):
    calls = []
    expected_by_input = {
        case["input"]: case["expected_output"]
        for question in load_question_bank()
        for case in question["test_cases"]
    }

    def validate_language(language):
        return "g++-15" if language == "cpp" else "python-3.14"

    async def execute(source_code, language, stdin):
        calls.append((source_code, language, stdin))
        return {
            "status": {"id": 3, "description": "Accepted"}, "status_id": 3,
            "stdout": expected_by_input[stdin] + "\n", "stderr": "", "compile_output": None,
            "time": "0.01",
            "memory": 512,
        }

    monkeypatch.setattr(routes.coding_execution_service, "validate_language", validate_language)
    monkeypatch.setattr(routes.coding_execution_service, "execute", execute)
    return calls


def start_attempt(client):
    response = client.post("/api/coding/attempt/start")
    assert response.status_code == 200, response.text
    return response.json()


def submission(attempt, question, code="solution", language="python", **extra):
    return {
        "attempt_id": attempt["attempt_id"],
        "question_id": question["id"],
        "language": language,
        "source_code": code,
        **extra,
    }


def test_dataset_has_twenty_questions_three_cases_and_question_lookup():
    questions = load_question_bank()
    assert len(questions) == 20
    assert all(len(question["test_cases"]) == 3 for question in questions)
    for question in questions:
        assert "TODO" in question["starter_code_cpp"]
        assert "int main()" in question["starter_code_cpp"]
        assert "cin >>" in question["starter_code_cpp"]
        assert "TODO" in question["starter_code_python"]
        assert "sys.stdin" in question["starter_code_python"]
        compile(question["starter_code_python"], f"question-{question['id']}", "exec")
        assert "First line:" in question["input_format"] or "One line" in question["input_format"]
        assert question["output_format"]
        assert "server-checked" not in question["output_format"]
    assert find_question(1)["title"] == "Two Sum"
    assert "n = data[0]" in find_question(1)["starter_code_python"]
    assert "target = data[1 + n]" in find_question(1)["starter_code_python"]
    assert "int target; cin >> target;" in find_question(1)["starter_code_cpp"]
    assert "long long a, b; cin >> a >> b;" in find_question(20)["starter_code_cpp"]
    assert find_question(99999) is None


def test_examples_are_derived_from_the_existing_first_test_case():
    for question in load_question_bank():
        public = evaluator.public_question(question)
        assert public["examples"] == [{
            "input": question["test_cases"][0]["input"],
            "output": question["test_cases"][0]["expected_output"],
        }]


def test_question_selection_excludes_only_the_previous_round(monkeypatch):
    calls = []

    class FixedRandom:
        def sample(self, population, count):
            calls.append(list(population))
            preferred = [3, 14, 5, 19]
            selected = [question_id for question_id in preferred if question_id in population]
            return selected[:count]

    monkeypatch.setattr(evaluator.secrets, "SystemRandom", FixedRandom)
    first = evaluator.select_question_ids()
    second = evaluator.select_question_ids(set(first))
    third = evaluator.select_question_ids(set(second))
    assert first == [3, 14]
    assert second == [5, 19]
    assert third == [3, 14]
    assert len(calls[0]) == 20 and len(calls[1]) == 18 and len(calls[2]) == 18


def test_question_selection_falls_back_when_fewer_than_two_are_eligible(monkeypatch):
    populations = []

    class FirstTwoRandom:
        def sample(self, population, count):
            populations.append(list(population))
            return list(population)[:count]

    monkeypatch.setattr(evaluator.secrets, "SystemRandom", FirstTwoRandom)
    selected = evaluator.select_question_ids(set(range(1, 20)))
    assert len(selected) == 2
    assert len(populations[0]) == 20


def test_dataset_errors_are_reported(monkeypatch, tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(evaluator, "DATASET_PATH", bad_file)
    with pytest.raises(CodingDatasetError):
        load_question_bank()


def test_active_question_allowlist_excludes_tests_and_solutions(client_and_sessions):
    client, _, _ = client_and_sessions
    page = client.get("/coding")
    assert page.status_code == 200
    assert "monaco-editor" in page.text
    assert "/api/coding" not in page.text
    question = client.get("/api/coding/questions/1")
    assert question.status_code == 200
    body = question.json()
    assert "reference_solution_cpp" not in body
    assert "reference_solution_python" not in body
    assert "test_cases" not in body
    assert "expected_output" not in json.dumps(body)
    assert body["examples"] == [{
        "input": find_question(1)["test_cases"][0]["input"],
        "output": find_question(1)["test_cases"][0]["expected_output"],
    }]
    assert 'id="question-examples"' in page.text
    assert client.get("/api/coding/questions").status_code == 200
    assert client.get("/api/coding/questions/99999").status_code == 404


def test_main_app_coding_page_and_attempt_start_create_exactly_two_questions(client_and_sessions):
    _, sessions, test_app = client_and_sessions
    main.app.dependency_overrides[main.get_db] = test_app.dependency_overrides[main.get_db]
    try:
        app_client = TestClient(main.app)
        app_client.cookies.set("user", "coding-user")
        page = app_client.get("/coding")
        assert page.status_code == 200
        assert 'id="monaco-editor"' in page.text
        assert 'href="/coding"' in page.text

        response = app_client.post("/api/coding/attempt/start", json={})
        assert response.status_code == 200
        attempt_id = response.json()["attempt_id"]
        assert response.json()["question_order"] == 1
        assert response.json()["question"]["id"]
        with sessions() as db:
            attempt = db.query(CodingAttempt).filter_by(id=attempt_id).one()
            rows = db.query(CodingQuestionResult).filter_by(coding_attempt_id=attempt_id).all()
            assert attempt.status == "in_progress"
            assert attempt.total_questions == 2
            assert len(rows) == 2
            assert len({row.question_id for row in rows}) == 2
    finally:
        main.app.dependency_overrides.pop(main.get_db, None)


def test_onlinecompiler_key_is_not_exposed_to_page_assets_or_coding_responses(
    client_and_sessions, monkeypatch, successful_onlinecompiler
):
    client, _, _ = client_and_sessions
    api_key = "private-onlinecompiler-key"
    monkeypatch.setattr("config.settings.settings.onlinecompiler_api_key", api_key)
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])

    page = client.get("/coding")
    question_response = client.get("/api/coding/questions/1")
    attempt = start_attempt(client)
    run_response = client.post("/api/coding/run", json=submission(attempt, attempt["question"]))

    assert page.status_code == question_response.status_code == run_response.status_code == 200
    assert api_key not in page.text
    assert api_key not in Path("static/coding/coding.js").read_text(encoding="utf-8")
    assert api_key not in question_response.text
    assert api_key not in run_response.text


def test_start_assigns_exactly_two_distinct_questions_and_no_third(client_and_sessions, monkeypatch):
    client, sessions, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    attempt = start_attempt(client)
    assert attempt["question_order"] == 1
    assert attempt["question"]["id"] == 1
    assert "reference_solution_cpp" not in attempt["question"]
    with sessions() as db:
        saved = db.query(CodingQuestionResult).filter_by(coding_attempt_id=attempt["attempt_id"]).all()
        assert len(saved) == 2
        assert len({row.question_id for row in saved}) == 2
        assert [row.question_order for row in sorted(saved, key=lambda row: row.question_order)] == [1, 2]
        assert all(row.total_test_cases == 3 for row in saved)
    third = client.post("/api/coding/run", json=submission(attempt, {"id": 3}))
    assert third.status_code == 409
    with sessions() as db:
        assert db.query(CodingQuestionResult).filter_by(coding_attempt_id=attempt["attempt_id"]).count() == 2


def test_run_then_submit_reuses_results_caps_execution_and_persists_report(
    client_and_sessions, monkeypatch, successful_onlinecompiler
):
    client, sessions, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    attempt = start_attempt(client)
    q1 = attempt["question"]

    # A forged expected output is ignored; the mock output is matched against the server dataset.
    first_run = client.post("/api/coding/run", json=submission(attempt, q1, expected_output="FORGED"))
    assert first_run.status_code == 200
    first_run_data = first_run.json()
    assert first_run_data["passed_test_cases"] == 3
    assert first_run_data["execution_count"] == 3
    assert first_run_data["attempt_execution_count"] == 3
    assert len(successful_onlinecompiler) == 3
    assert "expected_output" not in json.dumps(first_run_data)

    # Repeated Run shows cached output without making another provider call.
    cached_run = client.post("/api/coding/run", json=submission(attempt, q1))
    assert cached_run.status_code == 200
    assert "without another provider call" in cached_run.json()["message"]
    assert len(successful_onlinecompiler) == 3

    q1_submit = client.post("/api/coding/submit", json=submission(attempt, q1))
    assert q1_submit.status_code == 200
    assert q1_submit.json()["next"]["question"]["id"] == 2
    assert len(successful_onlinecompiler) == 3

    q2 = q1_submit.json()["next"]["question"]
    q2_submit = client.post("/api/coding/submit", json=submission(attempt, q2, language="cpp"))
    assert q2_submit.status_code == 200
    assert q2_submit.json()["completed"] is True
    assert q2_submit.json()["attempt_execution_count"] == 6
    assert len(successful_onlinecompiler) == 6

    with sessions() as db:
        saved_attempt = db.query(CodingAttempt).filter_by(id=attempt["attempt_id"]).one()
        saved_rows = db.query(CodingQuestionResult).filter_by(coding_attempt_id=saved_attempt.id).all()
        assert saved_attempt.status == "completed"
        assert saved_attempt.total_questions == 2
        assert saved_attempt.attempted_questions == 2
        assert saved_attempt.solved_questions == 2
        assert saved_attempt.total_test_cases == 6
        assert saved_attempt.passed_test_cases == 6
        assert saved_attempt.overall_score == 100.0
        assert len(saved_rows) == 2
        assert sum(row.execution_count for row in saved_rows) == 6
        assert all(len(row.test_results) == 3 for row in saved_rows)

    report = client.get(f"/api/coding/attempt/{attempt['attempt_id']}/result")
    assert report.status_code == 200
    report_body = report.json()
    assert report_body["overall_score"] == 100.0
    assert len(report_body["questions"]) == 2
    assert "reference_solution" in report_body["questions"][0]
    assert "expected_output" not in json.dumps(report_body)
    report_page = client.get(f"/coding/attempt/{attempt['attempt_id']}/result")
    assert report_page.status_code == 200
    assert "intervista-sidebar" in report_page.text
    assert "intervista-main" in report_page.text
    assert "app-page-header" in report_page.text
    assert "app-panel report-hero" in report_page.text
    assert "/static/profile-dashboard.css" in report_page.text

    # A completed attempt cannot add or execute a third question.
    blocked = client.post("/api/coding/submit", json=submission(attempt, {"id": 3}))
    assert blocked.status_code == 404
    assert len(successful_onlinecompiler) == 6


def test_execution_limit_is_server_enforced_and_different_code_cannot_reuse_results(
    client_and_sessions, monkeypatch, successful_onlinecompiler
):
    client, sessions, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    attempt = start_attempt(client)
    question = attempt["question"]
    assert client.post("/api/coding/run", json=submission(attempt, question)).status_code == 200
    changed_code = client.post("/api/coding/run", json=submission(attempt, question, code="changed"))
    assert changed_code.status_code == 429
    assert "Execution limit reached" in changed_code.json()["detail"]
    assert len(successful_onlinecompiler) == 3
    with sessions() as db:
        saved_attempt = db.query(CodingAttempt).filter_by(id=attempt["attempt_id"]).one()
        row = db.query(CodingQuestionResult).filter_by(coding_attempt_id=saved_attempt.id, question_order=1).one()
        assert row.execution_count == 3
        assert saved_attempt.execution_count == 3


def test_authentication_protects_coding_page_and_api(client_and_sessions):
    client, _, _ = client_and_sessions
    client.cookies.clear()
    assert client.get("/coding").status_code == 401
    assert client.get("/api/coding/questions").status_code == 401
    assert client.post("/api/coding/attempt/start").status_code == 401


def test_interview_history_lists_only_completed_owned_coding_rounds(client_and_sessions):
    _, sessions, _ = client_and_sessions
    with sessions() as db:
        user = db.query(User).filter_by(username="coding-user").one()
        attempt = CodingAttempt(
            user_id=user.id, status="completed", total_questions=2,
            attempted_questions=2, solved_questions=1, total_test_cases=6,
            passed_test_cases=4, overall_score=66.67,
        )
        db.add(attempt)
        db.commit()
        attempt_id = attempt.id

    main.app.dependency_overrides[main.get_db] = client_and_sessions[0].app.dependency_overrides[main.get_db]
    history_client = TestClient(main.app)
    history_client.cookies.set("user", "coding-user")
    response = history_client.get("/interview-history")
    assert response.status_code == 200
    assert "Coding Rounds" in response.text
    assert f"/coding/attempt/{attempt_id}/result" in response.text
    assert '<a href="/coding"><i>⌘</i>Coding Round</a>' in response.text
    main.app.dependency_overrides.pop(main.get_db, None)


def test_each_main_sidebar_template_always_includes_coding_round():
    for path in (
        "templates/profile.html",
        "templates/index.html",
        "templates/interview_history.html",
        "templates/coding_round.html",
        "templates/coding_result.html",
    ):
        source = Path(path).read_text(encoding="utf-8")
        nav_start = source.index('<nav class="primary-nav">')
        nav_end = source.index("</nav>", nav_start)
        nav = source[nav_start:nav_end]
        assert 'href="/coding"' in nav, path
        assert "Coding Round" in nav, path


def test_start_attempt_avoids_only_the_previous_completed_round_questions(
    client_and_sessions, monkeypatch
):
    client, sessions, _ = client_and_sessions
    requested_exclusions = []

    def choose_ids(excluded_question_ids=None):
        excluded = set(excluded_question_ids or set())
        requested_exclusions.append(excluded)
        return ([1, 2], [3, 4], [1, 2])[len(requested_exclusions) - 1]

    monkeypatch.setattr(routes, "select_question_ids", choose_ids)
    first = start_attempt(client)
    with sessions() as db:
        attempt = db.query(CodingAttempt).filter_by(id=first["attempt_id"]).one()
        attempt.status = "completed"
        attempt.completed_at = datetime.utcnow()
        db.commit()

    second = start_attempt(client)
    with sessions() as db:
        attempt = db.query(CodingAttempt).filter_by(id=second["attempt_id"]).one()
        attempt.status = "completed"
        attempt.completed_at = datetime.utcnow()
        db.commit()

    third = start_attempt(client)
    assert requested_exclusions == [set(), {1, 2}, {3, 4}]
    assert first["question"]["id"] in {1, 2}
    assert second["question"]["id"] in {3, 4}
    assert third["question"]["id"] in {1, 2}


def test_incomplete_attempt_cannot_reveal_reference_solution(client_and_sessions, monkeypatch):
    client, _, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    attempt = start_attempt(client)
    report = client.get(f"/api/coding/attempt/{attempt['attempt_id']}/result")
    assert report.status_code == 404
    assert "reference_solution" not in report.text


def test_report_ownership_is_enforced(client_and_sessions, monkeypatch):
    client, sessions, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    attempt = start_attempt(client)
    with sessions() as db:
        db.add(User(username="other-user", password="not-used"))
        db.commit()
    client.cookies.set("user", "other-user")
    assert client.get(f"/api/coding/attempt/{attempt['attempt_id']}/result").status_code == 404


def test_output_comparison_and_scores_are_strict_but_ignore_outer_whitespace():
    assert normalize_output("\r\n 1 2 \r\n") == "1 2"
    accepted = evaluate_execution(1, {"status": {"id": 3, "description": "Accepted"}, "stdout": "0 1\r\n"}, "0 1")
    wrong = evaluate_execution(1, {"status": {"id": 3, "description": "Accepted"}, "stdout": "0  1"}, "0 1")
    assert accepted["passed"] is True
    assert wrong["status"] == "Wrong Answer"
    assert wrong["expected_output"] == "0 1"
    assert wrong["stdout"] == "0  1"
    assert score_tests([accepted, wrong, accepted]) == (2, 66.67)


def test_successful_execution_warnings_do_not_become_wrong_answer_reason():
    result = evaluate_execution(1, {
        "status": {"id": 3, "description": "Accepted"},
        "stdout": "Indices: [0, 1]",
        "stderr": "warning: unused variable",
    }, "0 1")
    assert result["status"] == "Wrong Answer"
    assert result["stderr"] == ""
    assert result["compiler_warning"] == "warning: unused variable"
    assert result["expected_output"] == "0 1"


def test_onlinecompiler_success_run_response_is_complete_for_frontend(
    client_and_sessions, monkeypatch, successful_onlinecompiler
):
    client, _, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    attempt = start_attempt(client)
    response = client.post("/api/coding/run", json=submission(attempt, attempt["question"]))
    assert response.status_code == 200
    payload = response.json()
    assert payload["passed_test_cases"] == 3
    assert [result["status"] for result in payload["results"]] == ["Accepted"] * 3
    assert payload["execution_count"] == 3
    assert len(successful_onlinecompiler) == 3


@pytest.mark.parametrize(
    ("status_id", "description", "expected"),
    [(5, "Time Limit Exceeded", "Time Limit Exceeded"),
     (6, "Compilation Error", "Compilation Error"),
     (11, "Runtime Error (NZEC)", "Runtime Error")],
)
def test_execution_statuses_map_to_clear_test_results(status_id, description, expected):
    result = evaluate_execution(2, {"status": {"id": status_id, "description": description}}, "output")
    assert result["status"] == expected
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("language", "expected_compiler", "expected_input"),
    [("python", "python-3.14", "stdin-python"), ("cpp", "g++-15", "stdin-cpp")],
)
def test_onlinecompiler_success_posts_one_test_with_server_auth(
    monkeypatch, language, expected_compiler, expected_input
):
    calls = []
    api_key = "server-only-test-key"

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            "output": "answer\n", "error": "", "status": "success", "exit_code": 0,
            "signal": None, "time": "0.02", "total": "0.03", "memory": "8192",
        })

    def client_factory(**kwargs):
        assert kwargs["timeout"].read <= 40
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("config.settings.settings.onlinecompiler_base_url", "https://api.onlinecompiler.io")
    monkeypatch.setattr("config.settings.settings.onlinecompiler_api_key", api_key)
    service = CodingExecutionService(client_factory=client_factory)
    result = asyncio.run(service.execute("print('answer')", language, expected_input))

    assert result["status_id"] == 3
    assert result["stdout"] == "answer\n"
    assert result["time"] == "0.02"
    assert result["memory"] == "8192"
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "POST"
    assert request.url == "https://api.onlinecompiler.io/api/run-code-sync/"
    assert request.headers["Authorization"] == api_key
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.read()) == {
        "compiler": expected_compiler,
        "code": "print('answer')",
        "input": expected_input,
    }
    assert api_key not in json.dumps(result)


@pytest.mark.parametrize(
    ("language", "payload", "expected_status"),
    [
        ("cpp", {"status": "error", "exit_code": 1, "output": "", "error": "main.cpp:1: error: expected ';'"}, "Compilation Error"),
        ("python", {"status": "error", "exit_code": 1, "output": "", "error": "Traceback (most recent call last): ValueError: bad input"}, "Runtime Error"),
        ("python", {"status": "error", "exit_code": 1, "output": "", "error": "SyntaxError: invalid syntax"}, "Compilation Error"),
        ("python", {"status": "error", "exit_code": 124, "output": "", "error": "Execution exceeded time limit"}, "Time Limit Exceeded"),
    ],
)
def test_onlinecompiler_execution_failures_map_to_existing_statuses(monkeypatch, language, payload, expected_status):
    def handler(request):
        return httpx.Response(200, json=payload)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("config.settings.settings.onlinecompiler_base_url", "https://api.onlinecompiler.io")
    monkeypatch.setattr("config.settings.settings.onlinecompiler_api_key", "test-key")
    result = asyncio.run(CodingExecutionService(client_factory=client_factory).execute("code", language, ""))
    evaluated = evaluate_execution(1, result, "expected")
    assert evaluated["status"] == expected_status
    assert evaluated["passed"] is False


def test_onlinecompiler_provider_and_network_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr("config.settings.settings.onlinecompiler_base_url", "https://api.onlinecompiler.io")
    monkeypatch.setattr("config.settings.settings.onlinecompiler_api_key", "secret-key")

    def provider_error(request):
        return httpx.Response(503, json={"detail": "private provider internals"})

    service = CodingExecutionService(client_factory=lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(provider_error), **kwargs
    ))
    with pytest.raises(CodingExecutionError, match="busy or unavailable") as caught:
        asyncio.run(service.execute("code", "python", ""))
    assert "secret-key" not in str(caught.value)
    assert "private provider internals" not in str(caught.value)

    def auth_error(request):
        return httpx.Response(401, json={"detail": "secret-key rejected"})

    auth_service = CodingExecutionService(client_factory=lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(auth_error), **kwargs
    ))
    with pytest.raises(CodingExecutionConfigurationError, match="authentication failed") as caught:
        asyncio.run(auth_service.execute("code", "python", ""))
    assert "secret-key" not in str(caught.value)

    def network_error(request):
        raise httpx.ConnectError("secret-key is not relevant here", request=request)

    network_service = CodingExecutionService(client_factory=lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(network_error), **kwargs
    ))
    with pytest.raises(CodingExecutionError, match="could not process"):
        asyncio.run(network_service.execute("code", "python", ""))

    timeout_calls = []

    def timeout_error(request):
        timeout_calls.append(request)
        raise httpx.ReadTimeout("provider request exceeded timeout", request=request)

    timeout_service = CodingExecutionService(client_factory=lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(timeout_error), **kwargs
    ))
    with pytest.raises(CodingExecutionError, match="timed out"):
        asyncio.run(timeout_service.execute("code", "python", ""))
    assert len(timeout_calls) == 1


def test_missing_onlinecompiler_configuration_does_not_consume_execution(client_and_sessions, monkeypatch):
    client, sessions, _ = client_and_sessions
    monkeypatch.setattr(routes, "select_question_ids", lambda excluded_question_ids=None: [1, 2])
    monkeypatch.setattr("config.settings.settings.onlinecompiler_api_key", "")
    attempt = start_attempt(client)
    response = client.post("/api/coding/run", json=submission(attempt, attempt["question"]))
    assert response.status_code == 503
    assert "authentication is not configured" in response.json()["detail"]
    with sessions() as db:
        saved = db.query(CodingAttempt).filter_by(id=attempt["attempt_id"]).one()
        row = db.query(CodingQuestionResult).filter_by(coding_attempt_id=saved.id, question_order=1).one()
        assert saved.execution_count == 0
        assert row.execution_count == 0

