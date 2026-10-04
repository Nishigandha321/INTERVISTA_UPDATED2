"""Regression coverage for answer continuation and repeat-session routing."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.responses import JSONResponse

import main


class _Request:
    def __init__(self, body=None):
        self._body = body or {}

    async def json(self):
        return self._body


class _AttemptDB:
    def __init__(self):
        self.next_id = 1

    def add(self, row):
        row.id = self.next_id
        self.next_id += 1

    def flush(self):
        pass

    def commit(self):
        pass


def test_two_submitted_answers_continue_with_answer_grounded_questions(monkeypatch):
    user = SimpleNamespace(id=17, username="continuity-test")
    session = {
        "session_id": "session-1", "interview_mode": "individual_practice",
        "interview_type": "technical", "role": "Backend Engineer", "level": "Junior",
        "current_question": "Why did you choose FastAPI?", "current_turn_id": "turn-1",
        "current_question_type": "main", "current_category": "technical",
        "current_topic": "technical", "current_difficulty": "Easy",
        "previous_action": "START", "main_question_count": 1, "probe_count": 0,
        "turn_count": 0, "qa_history": [], "answers": [], "recent_conversation": [],
        "questions": ["Why did you choose FastAPI?"], "asked_questions": ["Why did you choose FastAPI?"],
        "categories": ["technical"],
    }
    monkeypatch.setitem(main.interview_sessions, user.username, session)
    monkeypatch.setattr(main, "get_current_user", lambda request, db: user)
    monkeypatch.setattr(main, "_conversation_limits", lambda current: (5, 2, 12))

    async def grounded_question(_user, _db, current, action, answer, analysis):
        keyword = main.choose_focus_keyword(answer, analysis)
        return main._question_response(
            current, f"You mentioned {keyword}. How did you apply it?", "technical",
            action, answer, analysis,
        )

    monkeypatch.setattr(main, "_generate_conversation_question", grounded_question)
    db = _AttemptDB()

    async def submit(turn_id, answer_id, answer):
        response = await main.api_interview_respond(
            _Request({"session_id": "session-1", "turn_id": turn_id,
                     "answer_id": answer_id, "answer": answer, "duration": 14}), db
        )
        return json.loads(response.body)

    first = asyncio.run(submit("turn-1", "answer-1", "FastAPI gave us automatic validation."))
    assert first["is_final"] is False
    assert "FastAPI" in first["question"] or "validation" in first["question"].lower()
    assert session["qa_history"][0]["answer"] == "FastAPI gave us automatic validation."
    assert session["turn_count"] == 1

    session["current_turn_id"] = first["turn_id"]
    second_answer = "Pydantic schemas kept the request validation consistent."
    second = asyncio.run(submit(first["turn_id"], "answer-2", second_answer))
    assert second["is_final"] is False
    assert "Pydantic" in second["question"] or "schemas" in second["question"].lower()
    assert session["qa_history"][1]["answer"] == second_answer
    assert session["turn_count"] == 2
    assert len(session["questions"]) == 3


@pytest.mark.parametrize(
    ("mode", "interview_type", "expected_mode", "expected_round"),
    [
        ("individual", "technical", "individual_practice", "technical"),
        ("individual", "hr", "individual_practice", "hr"),
        ("full", "technical", "placement_simulation", "technical"),
    ],
)
def test_give_interview_again_restarts_source_mode_and_type(
    monkeypatch, mode, interview_type, expected_mode, expected_round
):
    user = SimpleNamespace(id=4, username="repeat-test")
    report = {"simulation_mode": mode, "interview_type": interview_type,
              "role": "Engineer", "level": "Mid"}
    source = SimpleNamespace(id=91, user_id=user.id, report_json=json.dumps(report), role="Engineer")

    class Query:
        def filter(self, *args):
            return self

        def first(self):
            return source

    class DB:
        def query(self, model):
            return Query()

    monkeypatch.setattr(main, "get_current_user", lambda request, db: user)
    started = {}

    def start_fresh(**kwargs):
        started.update(kwargs)
        fresh = {"session_id": "new-session", "questions": [], "answers": [], "qa_history": []}
        monkeypatch.setitem(main.interview_sessions, user.username, fresh)
        return JSONResponse({"started": True})

    monkeypatch.setattr(main, "_build_interview_context", start_fresh)
    response = main.repeat_interview(_Request(), 91, DB())

    assert response.status_code == 200
    assert started["interview_mode"] == expected_mode
    assert started["selected_round"] == expected_round
    assert main.interview_sessions[user.username]["session_id"] == "new-session"
    assert main.interview_sessions[user.username]["qa_history"] == []


def test_session_metadata_is_saved_for_repeat_routing():
    individual = main._report_session_metadata({
        "interview_mode": "individual_practice", "interview_type": "hr",
        "current_round": "hr", "session_id": "hr-session",
    })
    full = main._report_session_metadata({
        "interview_mode": "placement_simulation", "interview_type": "technical",
        "current_round": "technical", "session_id": "full-session",
    })
    assert (individual["simulation_mode"], individual["interview_type"]) == ("individual", "hr")
    assert (full["simulation_mode"], full["interview_type"]) == ("full", "technical")


def test_report_requires_finished_session_or_explicit_early_end_and_uses_server_answers():
    answers = [{"question": "Q1", "answer": "recorded answer"}]
    session = {"session_id": "session-1", "finished": False, "qa_history": answers}
    with pytest.raises(main.HTTPException) as error:
        main._authoritative_evaluation_answers(session, {"session_id": "session-1"})
    assert error.value.status_code == 409
    assert session["finished"] is False

    accepted = main._authoritative_evaluation_answers(session, {
        "session_id": "session-1", "end_interview": True,
        "questions_answers": [{"question": "spoofed", "answer": "client data"}],
    })
    assert session["finished"] is True
    assert accepted == answers

