import json
import asyncio
import re
from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base
from models import GroupDiscussionSession, User
from services.group_discussion.routes import _initial_state, create_group_discussion_router


class FakeLLM:
    settings = SimpleNamespace(groq_api_key="test-key")

    def __init__(self):
        self.ai_calls = 0
        self.report_calls = 0
        self.prompts = []

    async def invoke(self, prompt, **kwargs):
        if prompt.startswith("Evaluate only"):
            self.report_calls += 1
            return json.dumps({
                "communication": 84, "clarity": 81, "relevance": 86,
                "argument_strength": 78, "participation": 88,
                "response_to_others": 83,
                "strengths": ["You responded to prior points."],
                "areas_to_improve": ["Use a more specific supporting example."],
                "feedback_points": ["State one claim, give evidence, then connect it to the topic."],
                "topic_suggestions": ["Compare short-term and long-term effects.", "Address a counterargument."],
            })
        self.ai_calls += 1
        self.prompts.append(prompt)
        if self.ai_calls == 2:
            return "A long paragraph that must be shortened by the GD layer. It keeps going with extra wording.\nAnd this third line must never show up because the response limit is enforced."
        return f"AI contribution {self.ai_calls} responds to the discussion. It adds one useful point."


def test_complete_isolated_gd_flow_with_voice_and_persisted_report():
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with TestingSession() as db:
        user = User(username="gd-candidate", password="test")
        db.add(user)
        db.commit()
        user_id = user.id

    def get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    def get_current_user(request, db):
        if request.cookies.get("user") != "gd-candidate":
            return None
        return db.query(User).filter(User.id == user_id).first()

    llm = FakeLLM()
    app = FastAPI()
    app.include_router(create_group_discussion_router(
        get_current_user, get_db, llm,
        lambda request_args: SimpleNamespace(text="I would balance innovation with safeguards."),
    ))
    client = TestClient(app)
    client.cookies.set("user", "gd-candidate")

    assert client.get("/gd").status_code == 200
    started = client.post("/gd/start", data={
        "topic_mode": "custom", "custom_topic": "Should AI be regulated?",
        "total_participants": "3",
    }, follow_redirects=False)
    assert started.status_code == 303
    session_url = started.headers["location"]
    public_id = session_url.rsplit("/", 1)[-1]
    page = client.get(session_url)
    assert page.status_code == 200
    assert "data-public-id" in page.text
    assert "Your turn — you can start speaking" in page.text

    state = client.get(f"/api/gd/session/{public_id}").json()
    assert state["total_participants"] == 3
    assert 0 < state["total_seconds_remaining"] <= 300
    assert state["main_round_limit"] == 1
    assert state["planned_max_seconds"] == 269
    assert state["planned_max_seconds"] < 300
    assert [person["id"] for person in state["participants"]] == ["ai_1", "ai_2", "user"]

    def speak_ai_until_candidate(current):
        while current["current_speaker"]["id"].startswith("ai_"):
            stale_token = current["turn_token"]
            result = client.post(f"/api/gd/session/{public_id}/ai-turn", json={"turn_token": stale_token})
            assert result.status_code == 200
            current = result.json()
            entry = current["discussion_history"][-1]
            assert len(entry["content"].splitlines()) <= 3
            sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+", entry["content"]) if part.strip()])
            assert 2 <= sentence_count <= 3
            if llm.ai_calls == 1:
                duplicate = client.post(f"/api/gd/session/{public_id}/ai-turn", json={"turn_token": stale_token})
                assert duplicate.status_code == 409
                assert llm.ai_calls == 1
        return current

    state = speak_ai_until_candidate(state)
    transcript = client.post(
        "/api/gd/transcribe",
        files={"file": ("answer.webm", b"fake audio", "audio/webm")},
        data={"context_prompt": state["topic"]},
    )
    assert transcript.status_code == 200
    assert transcript.json()["transcript"].startswith("I would balance")
    state = client.post(f"/api/gd/session/{public_id}/respond", json={
        "transcript": transcript.json()["transcript"], "input_mode": "voice",
        "duration_seconds": 5, "turn_token": state["turn_token"],
    }).json()
    assert state["phase"] == "main_discussion"

    state = client.get(f"/api/gd/session/{public_id}").json()
    assert state["phase"] == "main_discussion"
    state = speak_ai_until_candidate(state)
    state = client.post(f"/api/gd/session/{public_id}/respond", json={
        "transcript": "I would add measurable outcomes and hear the counterpoint.",
        "input_mode": "text", "turn_token": state["turn_token"],
    }).json()
    assert state["phase"] == "conclusion"
    assert state["final_round_warning"] is True
    assert state["main_rounds_completed"] == 1

    state = speak_ai_until_candidate(state)
    state = client.post(f"/api/gd/session/{public_id}/respond", json={
        "transcript": "My closing view is to balance accountability with practical implementation.",
        "input_mode": "text", "turn_token": state["turn_token"],
    }).json()
    assert state["phase"] == "completed"
    assert 0 <= state["report"]["overall_score"] <= 100
    assert state["report"]["speaking_pace"]["measured"] is True
    assert llm.ai_calls == 6
    assert llm.report_calls == 1
    assert any("Use simple, natural spoken English" in prompt for prompt in llm.prompts)
    assert any("Most recent contribution to address" in prompt for prompt in llm.prompts)

    report_page = client.get(f"/gd/report/{public_id}")
    assert report_page.status_code == 200
    for expected in ("Performance report", "Speaking pace", "Response to others", "TOPIC-SPECIFIC IDEAS", "Actionable feedback"):
        assert expected in report_page.text
    assert client.get(f"/api/gd/session/{public_id}/report").json()["overall_score"] == state["report"]["overall_score"]
    assert client.get(session_url).status_code == 200


def test_incomplete_exit_and_missed_final_turn_do_not_generate_report():
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with TestingSession() as db:
        user = User(username="gd-exit", password="test")
        db.add(user)
        db.commit()
        user_id = user.id

    def get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    def get_current_user(request, db):
        if request.cookies.get("user") != "gd-exit":
            return None
        return db.query(User).filter(User.id == user_id).first()

    llm = FakeLLM()
    app = FastAPI()
    app.include_router(create_group_discussion_router(get_current_user, get_db, llm, lambda _: None))
    client = TestClient(app)
    client.cookies.set("user", "gd-exit")
    started = client.post("/gd/start", data={
        "topic_mode": "custom", "custom_topic": "Remote work", "total_participants": "3",
    }, follow_redirects=False)
    public_id = started.headers["location"].rsplit("/", 1)[-1]
    assert client.get(f"/gd/session/{public_id}").status_code == 200  # Exit remains a plain page navigation.
    assert client.get(f"/gd/report/{public_id}", follow_redirects=False).status_code == 303
    assert llm.report_calls == 0

    with TestingSession() as db:
        record = db.query(GroupDiscussionSession).filter_by(public_id=public_id).one()
        state = json.loads(record.state_json)
        state["phase"] = "conclusion"
        state["current_turn_index"] = state["speaker_order"].index("user")
        state["current_speaker"] = "user"
        state["turn_deadline_at"] = datetime(2000, 1, 1).astimezone().isoformat()
        record.state_json = json.dumps(state)
        db.commit()

    ended = client.get(f"/api/gd/session/{public_id}").json()
    assert ended["phase"] == "expired"
    assert ended["current_speaker"] is None
    assert client.get(f"/gd/report/{public_id}", follow_redirects=False).status_code == 303
    assert llm.report_calls == 0


def test_gd_participant_options_are_exact_total_counts():
    for count in (3, 4, 5):
        state = _initial_state("topic", count)
        assert len(state["participants"]) == count
        assert state["participants"][-1]["id"] == "user"
        assert len(state["speaker_order"]) == count
