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


def test_multi_question_evaluations_keep_question_identity_and_all_fields(monkeypatch):
    class EvaluationChain:
        async def invoke(self, payload, **kwargs):
            question = payload["question"]
            return SimpleNamespace(
                status="success",
                output=json.dumps({
                    "score": 82, "relevance_score": 80, "explanation_depth_score": 78,
                    "star_method_score": 70, "structured_thinking_score": 82,
                    "problem_solving_score": 80, "strengths": [f"Relevant answer for {question}"],
                    "weaknesses": [f"Add an example for {question}"],
                    "ideal_answer": f"A concise ideal answer for {question}.",
                    "weak_topics": [], "C": 0.8, "K": 0.8, "F": 0.8, "S": 0.8,
                }),
            )

    monkeypatch.setattr(main, "evaluation_chain", EvaluationChain())
    questions = [
        {"_attempt_id": 101, "question": "Explain Java inheritance.", "answer": "Java supports inheritance."},
        {"_attempt_id": 102, "question": "Explain SQL indexes.", "answer": "Indexes speed up database lookups."},
        {"_attempt_id": 103, "question": "Explain API validation.", "answer": "The API validates each request."},
    ]
    result = asyncio.run(main.evaluate_content("Engineer", "Junior", questions))
    evaluations = result["answers"]
    assert len(evaluations) == len(questions)
    reordered = list(reversed(evaluations))
    for index, question in enumerate(questions):
        evaluation = main._evaluation_for_answer(question, index, reordered)
        assert evaluation["attempt_id"] == question["_attempt_id"]
        assert evaluation["question"] == question["question"]
        assert evaluation["candidate_answer"] == question["answer"]
        assert evaluation["evaluation_available"] is True
        assert evaluation["strengths"]
        assert evaluation["weaknesses"]
        assert evaluation["ideal_answer"]


def test_report_summary_uses_answer_facts_and_stays_within_three_sentences():
    report = {
        "overall_score": 76,
        "content_analysis": {"average_score": 76},
        "detailed_answers": [
            {
                "question": "Explain the API design.", "transcript": "I used FastAPI and JWT.",
                "evaluation_available": True, "score": 82,
                "strengths": ["Explained the API security choice clearly."],
                "weaknesses": ["Add a concrete example of failure handling."],
                "weak_topics": ["JWT"],
            },
            {
                "question": "How did you test it?", "transcript": "I ran the tests.",
                "evaluation_available": True, "score": 70,
                "strengths": ["Kept testing focused."],
                "weaknesses": ["Explain what the test results showed."],
                "weak_topics": [],
            },
        ],
    }
    summary = asyncio.run(main.generate_performance_summary(report))
    sentence_count = len([part for part in summary.split(".") if part.strip()])
    assert sentence_count <= 3
    assert "Explained the API security choice clearly" in summary
    assert "failure handling" in summary


def test_job_skill_findings_use_evaluation_strengths_and_weaknesses():
    demonstrated, work_on = main._interview_skill_findings([{
        "question": "How did you secure the API?",
        "transcript": "I used JWT to protect each endpoint.",
        "topic": "technical", "evaluation_available": True, "score": 84,
        "strengths": ["Clear explanation of JWT-based authorization."],
        "weaknesses": ["Discuss token expiry and failure handling."],
        "weak_topics": ["JWT"],
    }])
    assert "JWT" in demonstrated
    assert "Discuss token expiry and failure handling" in work_on

