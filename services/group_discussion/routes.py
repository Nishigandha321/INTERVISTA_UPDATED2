"""Authenticated routes and deterministic orchestrator for Group Discussion."""

import asyncio
import json
import logging
import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from models import GroupDiscussionSession

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

TOPICS = [
    "Should artificial intelligence be regulated?",
    "Remote work versus office work",
    "The role of social media in society",
    "Is a four-day work week practical?",
    "Technology and the future of education",
    "Sustainable growth versus rapid industrialisation",
    "Should college education focus more on practical skills?",
    "The impact of automation on employment",
    "Privacy and convenience in the digital age",
    "Should companies prioritise experience or potential?",
    "Can cities become fully sustainable?",
    "The benefits and risks of a cashless economy",
]

PERSONAS = [
    {"id": "ai_1", "name": "Analytical", "style": "make a logical point and explain the reason in plain language"},
    {"id": "ai_2", "name": "Contrarian", "style": "politely question the last point and offer a simple counterpoint"},
    {"id": "ai_3", "name": "Collaborative", "style": "build on the last speaker's idea and add a practical thought"},
    {"id": "ai_4", "name": "Assertive", "style": "state a confident, direct view and one clear next step"},
]
MAX_SESSION_SECONDS = 5 * 60
USER_TURN_SECONDS = 36
AI_RESPONSE_TIMEOUT_SECONDS = 15
REPORT_TIMEOUT_SECONDS = 30
AI_SPEECH_MAX_SECONDS = 8
MAIN_DISCUSSION_ROUNDS = 1
DISCUSSION_PHASE_COUNT = 3  # Opening, one main round, and one final round.
# Worst-case schedule for five total participants: 12 AI turns and three user turns.
# It leaves a small buffer for request, database, and browser overhead inside five minutes.
PLANNED_MAX_SECONDS = DISCUSSION_PHASE_COUNT * (
    len(PERSONAS) * (AI_RESPONSE_TIMEOUT_SECONDS + AI_SPEECH_MAX_SECONDS) + USER_TURN_SECONDS
)
_session_locks: dict[str, asyncio.Lock] = {}


def _now():
    return datetime.now(timezone.utc)


def _stamp(value=None):
    return (value or _now()).isoformat()


def _initial_state(topic: str, total_participants: int) -> dict:
    ai_count = total_participants - 1
    participants = [dict(p) for p in PERSONAS[:ai_count]] + [{"id": "user", "name": "You", "persona": "Candidate"}]
    started_at = _now()
    return {
        "topic": topic,
        "total_participants": total_participants,
        "participants": participants,
        "speaker_order": [p["id"] for p in participants],
        "used_fallback_angles": [],
        "phase": "opening",
        "round": 1,
        "current_turn_index": 0,
        "current_speaker": participants[0]["id"],
        "discussion_history": [],
        "started_at": _stamp(started_at),
        "deadline_at": _stamp(started_at + timedelta(seconds=MAX_SESSION_SECONDS)),
        "main_started_at": None,
        "main_round_limit": MAIN_DISCUSSION_ROUNDS,
        "main_rounds_completed": 0,
        "planned_max_seconds": PLANNED_MAX_SECONDS,
        "turn_deadline_at": None,
        "final_round_warning": False,
        "ended_at": None,
    }


def _loads(record: GroupDiscussionSession) -> dict:
    state = json.loads(record.state_json)
    if not state.get("deadline_at"):
        try:
            started = datetime.fromisoformat(state["started_at"])
            state["deadline_at"] = _stamp(started + timedelta(seconds=MAX_SESSION_SECONDS))
        except (KeyError, TypeError, ValueError):
            state["deadline_at"] = _stamp(_now() + timedelta(seconds=MAX_SESSION_SECONDS))
    state.setdefault("main_round_limit", MAIN_DISCUSSION_ROUNDS)
    state.setdefault("main_rounds_completed", 0)
    state.setdefault("used_fallback_angles", [])
    if (
        state.get("phase") not in {"completed", "expired"}
        and state.get("current_speaker") == "user"
        and not state.get("turn_deadline_at")
    ):
        state["turn_deadline_at"] = _stamp(
            _now() + timedelta(
                seconds=USER_TURN_SECONDS + AI_SPEECH_MAX_SECONDS
            )
        )
    return state


def _save(db: Session, record: GroupDiscussionSession, state: dict):
    record.state_json = json.dumps(state, ensure_ascii=False)
    record.status = state["phase"] if state["phase"] in {"completed", "expired"} else "active"
    record.updated_at = _now().replace(tzinfo=None)
    db.commit()


def _get_record(db: Session, public_id: str, user):
    record = db.query(GroupDiscussionSession).filter(
        GroupDiscussionSession.public_id == public_id,
        GroupDiscussionSession.user_id == user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Group discussion session not found")
    return record


def _speaker(state: dict) -> dict:
    speaker_id = state["speaker_order"][state["current_turn_index"]]
    return next(p for p in state["participants"] if p["id"] == speaker_id)


def _public_state(state: dict) -> dict:
    result = dict(state)
    result["current_speaker"] = _speaker(state) if state["phase"] not in {"completed", "expired"} else None
    result["turn_token"] = f"{len(state['discussion_history'])}:{state['phase']}:{state['round']}:{state['current_turn_index']}"
    result["total_seconds_remaining"] = _seconds_remaining(state.get("deadline_at"))
    result["turn_seconds_remaining"] = _seconds_remaining(state.get("turn_deadline_at")) if state.get("turn_deadline_at") else None
    return result


def _seconds_remaining(deadline):
    if not deadline:
        return None
    try:
        return max(0, math.ceil((datetime.fromisoformat(deadline) - _now()).total_seconds()))
    except (TypeError, ValueError):
        return 0


def _expire_if_due(state: dict) -> bool:
    """Advance an overdue user turn; never mark an incomplete session complete."""
    if state["phase"] in {"completed", "expired"}:
        return False
    if _seconds_remaining(state.get("deadline_at")) == 0:
        state["phase"] = "expired"
        state["ended_at"] = _stamp()
        state["current_speaker"] = None
        state["turn_deadline_at"] = None
        return True
    if state.get("current_speaker") == "user" and _seconds_remaining(state.get("turn_deadline_at")) == 0:
        # A missed final user turn is incomplete and must not create a report.
        if state["phase"] == "conclusion":
            state["phase"] = "expired"
            state["ended_at"] = _stamp()
            state["current_speaker"] = None
            state["turn_deadline_at"] = None
        else:
            _advance_after_contribution(state)
        return True
    return False


def _advance_after_contribution(state: dict):
    state["turn_deadline_at"] = None
    size = len(state["speaker_order"])
    next_index = state["current_turn_index"] + 1
    if state["phase"] == "conclusion":
        if next_index >= size:
            state["phase"] = "completed"
            state["ended_at"] = _stamp()
            state["current_turn_index"] = 0
            state["current_speaker"] = None
        else:
            state["current_turn_index"] = next_index
            state["current_speaker"] = state["speaker_order"][next_index]
            if state["current_speaker"] == "user":
               state["turn_deadline_at"] = _stamp(
                   _now() + timedelta(
                       seconds=USER_TURN_SECONDS + AI_SPEECH_MAX_SECONDS
                   )
               )
        return

    if next_index < size:
        state["current_turn_index"] = next_index
        state["current_speaker"] = state["speaker_order"][next_index]
        if state["current_speaker"] == "user":
          state["turn_deadline_at"] = _stamp(
              _now() + timedelta(
                  seconds=USER_TURN_SECONDS + AI_SPEECH_MAX_SECONDS
              )
            )
        return

    state["current_turn_index"] = 0
    state["current_speaker"] = state["speaker_order"][0]
    state["round"] += 1
    if state["phase"] == "opening":
        state["phase"] = "main_discussion"
        state["main_started_at"] = _stamp()
        return
    if state["phase"] == "main_discussion":
        state["main_rounds_completed"] += 1
        if state["main_rounds_completed"] >= state["main_round_limit"]:
            state["phase"] = "conclusion"
            state["final_round_warning"] = True
            if state["speaker_order"][0] == "user":
               state["turn_deadline_at"] = _stamp(
                   _now() + timedelta(
                       seconds=USER_TURN_SECONDS + AI_SPEECH_MAX_SECONDS
                   )
               )


def _short_ai_contribution(value: str) -> str:
    text = re.sub(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s*", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().strip('"“”')
    return text.replace("\n", " ").strip()

FALLBACK_ANGLES = [
    (
        "impact",
        "We should also consider how this issue affects people in their everyday lives, not just organisations or policy makers."
    ),
    (
        "economic",
        "Another angle is the economic impact, especially who gains from this change and who may face new costs."
    ),
    (
        "practicality",
        "It is also important to ask whether this idea is practical to implement consistently in real situations."
    ),
    (
        "risk",
        "We should look at the possible unintended consequences as well, because solving one problem can sometimes create another."
    ),
    (
        "long_term",
        "A long-term perspective is useful here because the effects may be very different after several years."
    ),
    (
        "fairness",
        "We should also think about fairness and whether different groups would experience the same change in the same way."
    ),
    (
        "evidence",
        "A strong way to judge this point would be to look at measurable results rather than relying only on assumptions."
    ),
    (
        "counterargument",
        "There is also a reasonable counterargument here, so considering the opposite view could make the discussion more balanced."
    ),
]


def _fallback_ai_contribution(state: dict, persona: dict) -> str:
    """Create a deterministic but non-repetitive GD contribution."""

    used = set(state.get("used_fallback_angles", []))

    selected = None
    for angle, template in FALLBACK_ANGLES:
        if angle not in used:
            selected = (angle, template)
            break

    # If every angle has been used, recycle deterministically rather than
    # repeating the previous contribution verbatim.
    if selected is None:
        index = len(used) % len(FALLBACK_ANGLES)
        selected = FALLBACK_ANGLES[index]

    angle, base = selected
    used.add(angle)
    state["used_fallback_angles"] = list(used)

    persona_name = persona.get("name", "")

    if persona_name == "Analytical":
        return (
            f"{base} "
            f"That helps us evaluate the topic using more than one dimension."
        )

    if persona_name == "Contrarian":
        return (
            f"{base} "
            f"I think this side deserves attention before we reach a firm conclusion."
        )

    if persona_name == "Collaborative":
        return (
            f"{base} "
            f"Adding this perspective could make the group's discussion more complete."
        )

    if persona_name == "Assertive":
        return (
            f"{base} "
            f"So I would include this factor before deciding on the final position."
        )

    return base


def _append_contribution(state: dict, speaker: dict, content: str, kind="speech"):
    entry = {
        "speaker_id": speaker["id"],
        "speaker": speaker["name"],
        "persona": speaker.get("name") if speaker["id"].startswith("ai_") else None,
        "phase": state["phase"],
        "round": state["round"],
        "kind": kind,
        "content": content.strip(),
        "created_at": _stamp(),
    }
    if speaker["id"] == "user":
        try:
            duration = float(kind.get("duration_seconds")) if isinstance(kind, dict) and kind.get("duration_seconds") else 0
        except (TypeError, ValueError):
            duration = 0
        input_mode = kind.get("input_mode", "text") if isinstance(kind, dict) else str(kind)
        entry["kind"] = input_mode
        entry["voice_metrics"] = _analyze_gd_speech(content, duration) if input_mode == "voice" and duration > 0 else None
    state["discussion_history"].append(entry)
    _advance_after_contribution(state)


def _analyze_gd_speech(text: str, duration_seconds: float) -> dict:
    """Small local metrics for GD voice contributions; does not load or alter interview STT."""
    words = text.split()
    word_count = len(words)
    duration = max(0.1, float(duration_seconds))
    pace = round(word_count * 60 / duration, 1)
    fillers = {"um", "uh", "uhm", "like", "actually", "basically", "literally", "you know"}
    lowered = text.casefold()
    filler_count = sum(lowered.count(filler) for filler in fillers)
    filler_rate = filler_count / max(word_count, 1)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    average_sentence_words = word_count / max(len(sentences), 1)
    clarity = max(0, min(100, round(100 - filler_rate * 150 - max(0, average_sentence_words - 30))))
    filler_confidence = max(0, 100 - filler_rate * 300)
    pace_confidence = 100 if 120 <= pace <= 160 else 70
    length_confidence = 90 if word_count > 60 else 60
    confidence = round(max(0, min(100, .4 * filler_confidence + .3 * pace_confidence + .3 * length_confidence)))
    return {
        "word_count": word_count,
        "duration_seconds": round(duration, 1),
        "speaking_pace_wpm": pace,
        "filler_word_count": filler_count,
        "clarity_score": clarity,
        "confidence_fluency": confidence,
    }


def _extract_json(raw: str) -> dict:
    value = str(raw or "").strip()

    if not value:
        raise ValueError("GD report response was empty")

    # Remove markdown code fences if present
    if "```" in value:
        value = value.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    # Find the JSON object even if the model added text before/after it
    start = value.find("{")
    end = value.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(
            f"GD report response did not contain JSON. Raw response: {value[:500]}"
        )

    json_text = value[start:end + 1]

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"GD report JSON could not be parsed: {exc}. "
            f"Raw response: {json_text[:500]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("GD report response was not a JSON object")

    return parsed


def _score(value, default=0):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _speech_summary(user_entries: list[dict]) -> dict:
    samples = [entry["voice_metrics"] for entry in user_entries if isinstance(entry.get("voice_metrics"), dict)]
    if not samples:
        return {"measured": False, "speaking_pace_wpm": None, "clarity": None, "confidence_fluency": None, "word_count": sum(len(e.get("content", "").split()) for e in user_entries), "voice_turns": 0}
    duration = sum(float(sample.get("duration_seconds") or 0) for sample in samples)
    words = sum(int(sample.get("word_count") or 0) for sample in samples)
    pace = round(words / max(duration / 60, 0.01), 1)
    clarity = round(sum(_score(sample.get("clarity_score")) for sample in samples) / len(samples))
    confidence_values = [sample.get("confidence_fluency") for sample in samples if sample.get("confidence_fluency") is not None]
    return {
        "measured": True,
        "speaking_pace_wpm": pace,
        "clarity": clarity,
        "confidence_fluency": round(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
        "word_count": words,
        "voice_turns": len(samples),
    }


async def _build_report(state: dict, llm_service) -> dict:
    user_entries = [entry for entry in state["discussion_history"] if entry.get("speaker_id") == "user"]
    speech = _speech_summary(user_entries)
    discussion = "\n".join(
        f"{entry['speaker']} ({entry.get('phase')}): {entry.get('content', '')[:700]}"
        for entry in state["discussion_history"]
    )
    user_count = len(user_entries)
    prompt = (
        "Evaluate only the candidate's actual group-discussion contributions using the shared conversation. "
        "Be evidence-based and concise; do not infer audio confidence or pace from text. "
        "Return ONLY one valid JSON object. Do not use Markdown. Do not use code fences. "
        "Do not include any explanation before or after the JSON. "
        "All score fields must be numbers from 0 to 100. "
        "fields communication, clarity, relevance, argument_strength, participation, response_to_others, plus arrays "
        "strengths (2-4), areas_to_improve (1-4), feedback_points (at most 3 actionable items), and topic_suggestions (2-4 "
        "specific useful points the candidate could have raised for this topic). Do not reward claims that are absent. "
        f"\nTopic: {state['topic']}\nCandidate contribution count: {user_count} of {state['round']} discussion rounds."
        f"\nDiscussion transcript:\n{discussion}\n"
    )
    evaluation = None
    used_llm = False
    try:
        raw = await asyncio.wait_for(
            llm_service.invoke(
                prompt,
                temperature=0.1,
                max_tokens=1400,
                use_cache=False,
                json_mode=True,
                response_format={"type": "json_object"},
                reasoning_effort="none",
                request_timeout_seconds=REPORT_TIMEOUT_SECONDS,
                min_generation_tokens=400,
            ),
            timeout=REPORT_TIMEOUT_SECONDS,
        )
        evaluation = _extract_json(raw)
        required_scores = {
            "communication", "clarity", "relevance", "argument_strength",
            "participation", "response_to_others",
        }
        if not required_scores.issubset(evaluation):
            raise ValueError("GD report response omitted required score fields")
        used_llm = True
    except Exception as exc:
        logger.warning("GD report evaluation unavailable: type=%s error=%s", type(exc).__name__, exc)

    if evaluation is None:
        avg_words = speech["word_count"] / max(user_count, 1)
        objective = speech["clarity"] if speech["clarity"] is not None else max(35, min(80, int(avg_words * 1.4)))
        evaluation = {
            "communication": objective,
            "clarity": objective,
            "relevance": 50,
            "argument_strength": 50,
            "participation": min(100, max(45, round(60 + min(avg_words, 80) * 0.5))),
            "response_to_others": 50,
            "strengths": ["You contributed throughout the discussion." if user_count >= 3 else "Your recorded contributions are available for review."],
            "areas_to_improve": ["Qualitative topic and argument evaluation was unavailable for this session."],
            "feedback_points": ["Review the transcript and add a concise example or supporting reason to each key point."],
            "topic_suggestions": [f"Consider framing the central trade-off in {state['topic']}.", "Support a position with a concrete example and acknowledge a credible counterpoint."],
        }

    result = {
        "topic": state["topic"],
        "overall_score": 0,
        "communication": _score(evaluation.get("communication")),
        "clarity": speech["clarity"] if speech["clarity"] is not None else _score(evaluation.get("clarity")),
        "speaking_pace": {
            "wpm": speech["speaking_pace_wpm"],
            "score": (100 if 110 <= speech["speaking_pace_wpm"] <= 170 else max(35, 100 - abs(speech["speaking_pace_wpm"] - 140) // 2)) if speech["measured"] else None,
            "measured": speech["measured"],
        },
        "relevance": _score(evaluation.get("relevance")),
        "argument_strength": _score(evaluation.get("argument_strength")),
        "participation": _score(evaluation.get("participation")),
        "response_to_others": _score(evaluation.get("response_to_others")),
        "confidence_fluency": speech["confidence_fluency"],
        "confidence_fluency_measured": speech["measured"],
        "voice_turns_measured": speech["voice_turns"],
        "strengths": _string_list(evaluation.get("strengths"), 4),
        "areas_to_improve": _string_list(evaluation.get("areas_to_improve"), 4),
        "feedback_points": _string_list(evaluation.get("feedback_points"), 3),
        "topic_suggestions": _string_list(evaluation.get("topic_suggestions"), 4),
        "candidate_turns": user_count,
        "evaluation_source": "llm" if used_llm else "estimated",
        "generated_at": _stamp(),
    }
    components = [
        (result["communication"], 15), (result["clarity"], 15),
        (result["relevance"], 15), (result["argument_strength"], 15),
        (result["participation"], 10), (result["response_to_others"], 15),
    ]
    if result["speaking_pace"]["score"] is not None:
        components.append((result["speaking_pace"]["score"], 10))
    if result["confidence_fluency"] is not None:
        components.append((result["confidence_fluency"], 5))
    result["overall_score"] = round(sum(score * weight for score, weight in components) / sum(weight for _, weight in components))
    return result


def _string_list(value, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:350] for item in value if str(item).strip()][:limit]


def _auth_user(request: Request, db: Session, get_current_user):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user


def create_group_discussion_router(get_current_user, get_db, llm_service, transcribe_audio):
    router = APIRouter()

    @router.get("/gd", response_class=HTMLResponse)
    def gd_setup(request: Request, db: Session = Depends(get_db)):
        user = get_current_user(request, db)
        if not user:
            return RedirectResponse("/login", status_code=303)
        return templates.TemplateResponse(request, "gd_setup.html", {
            "request": request, "username": user.username, "topics": TOPICS,
        })

    @router.post("/gd/start")
    def gd_start(
        request: Request,
        topic_mode: str = Form(...),
        custom_topic: str = Form(""),
        selected_topic: str = Form(""),
        total_participants: int = Form(...),
        db: Session = Depends(get_db),
    ):
        user = get_current_user(request, db)
        if not user:
            return RedirectResponse("/login", status_code=303)
        if total_participants not in {3, 4, 5}:
            raise HTTPException(status_code=422, detail="Choose 3, 4, or 5 total participants")
        if topic_mode == "custom":
            topic = " ".join(custom_topic.split())[:240]
        elif topic_mode == "random":
            topic = random.choice(TOPICS)
        else:
            raise HTTPException(status_code=422, detail="Choose a custom or random topic")
        if not topic:
            raise HTTPException(status_code=422, detail="Enter a discussion topic")
        state = _initial_state(topic, total_participants)
        public_id = str(uuid4())
        db.add(GroupDiscussionSession(
            public_id=public_id, user_id=user.id,
            state_json=json.dumps(state, ensure_ascii=False), status="active",
        ))
        db.commit()
        return RedirectResponse(f"/gd/session/{public_id}", status_code=303)

    @router.get("/gd/session/{public_id}", response_class=HTMLResponse)
    def gd_simulation(public_id: str, request: Request, db: Session = Depends(get_db)):
        user = get_current_user(request, db)
        if not user:
            return RedirectResponse("/login", status_code=303)
        record = _get_record(db, public_id, user)
        return templates.TemplateResponse(request, "gd_session.html", {
            "request": request, "username": user.username, "public_id": public_id,
            "initial_state": _public_state(_loads(record)),
        })

    @router.get("/api/gd/session/{public_id}")
    def gd_state(public_id: str, request: Request, db: Session = Depends(get_db)):
        user = _auth_user(request, db, get_current_user)
        record = _get_record(db, public_id, user)
        state = _loads(record)
        if _expire_if_due(state):
            _save(db, record, state)
        return _public_state(state)

    @router.get("/gd/report/{public_id}", response_class=HTMLResponse)
    async def gd_report(public_id: str, request: Request, db: Session = Depends(get_db)):
        user = get_current_user(request, db)
        if not user:
            return RedirectResponse("/login", status_code=303)
        record = _get_record(db, public_id, user)
        state = _loads(record)
        if state["phase"] != "completed":
            return RedirectResponse(f"/gd/session/{public_id}", status_code=303)
        if not state.get("report"):
            state["report"] = await _build_report(state, llm_service)
            _save(db, record, state)
        return templates.TemplateResponse(request, "gd_report.html", {
            "request": request, "username": user.username, "public_id": public_id,
            "state": state, "report": state["report"],
        })

    @router.get("/api/gd/session/{public_id}/report")
    def gd_report_data(public_id: str, request: Request, db: Session = Depends(get_db)):
        user = _auth_user(request, db, get_current_user)
        state = _loads(_get_record(db, public_id, user))
        if state["phase"] != "completed" or not state.get("report"):
            raise HTTPException(status_code=409, detail="The GD report is available after the session concludes")
        return state["report"]

    @router.post("/api/gd/session/{public_id}/ai-turn")
    async def gd_ai_turn(public_id: str, request: Request, payload: dict, db: Session = Depends(get_db)):
        user = _auth_user(request, db, get_current_user)
        lock = _session_locks.setdefault(public_id, asyncio.Lock())
        async with lock:
            record = _get_record(db, public_id, user)
            state = _loads(record)
            if _expire_if_due(state):
                _save(db, record, state)
            if state["phase"] in {"completed", "expired"}:
                raise HTTPException(status_code=409, detail="This discussion has ended")
            expected_token = str(payload.get("turn_token", ""))
            speaker = _speaker(state)
            if expected_token != _public_state(state)["turn_token"]:
                raise HTTPException(status_code=409, detail="This turn has already advanced; refresh the discussion")
            if not speaker or not speaker["id"].startswith("ai_"):
                raise HTTPException(status_code=409, detail="It is not an AI participant's turn")
            phase = state["phase"]
            recent = state["discussion_history"][-12:]
            previous = recent[-1]["content"] if recent else ""
            history_text = "\n".join(f"{entry['speaker']}: {entry['content'][:180]}" for entry in recent[-8:])
            persona = next(p for p in PERSONAS if p["id"] == speaker["id"])
            instruction = "Open with one clear point about the topic." if not recent else (
                "Refer to the previous speaker's point, then add one useful idea or reason."
            )
            if phase == "conclusion":
                instruction = "Refer to one point from the discussion and give your short closing view."
            prompt = (
                "You are one participant in a college or job interview group discussion. Keep your assigned personality. "
                "Use simple, natural spoken English. Avoid jargon, academic wording, and long sentences. "
                "Speak in 2 or 3 concise sentences, using 1 to 3 short lines; never use more than 3 sentences. "
                "Stay specific to the topic and respond to the previous speaker; do not give a generic statement. "
                f"\nTopic: {state['topic']}\nYour persona: {persona['name']} — {persona['style']}."
                f"\nPhase: {phase}; round: {state['round']}.\nInstruction: {instruction}"
                f"\nRecent shared discussion:\n{history_text or '(No one has spoken yet.)'}"
                f"\nMost recent contribution to address: {previous[:220] if previous else '(Opening turn.)'}"
                "\nReturn only the spoken contribution."
            )
            try:
                response = await asyncio.wait_for(
                    llm_service.invoke(
                        prompt,
                        temperature=0.45,
                        max_tokens=300,
                        use_cache=False,
                        reasoning_effort="none",
                        request_timeout_seconds=AI_RESPONSE_TIMEOUT_SECONDS,
                        min_generation_tokens=150,
                    ),
                    timeout=AI_RESPONSE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning(
                    "GD AI turn failed: session=%s type=%s",
                    public_id,
                    type(exc).__name__,
                )
            
                # The LLM failed, so use a deterministic fallback that introduces
                # a NEW discussion angle instead of paraphrasing the previous point.
                response = _fallback_ai_contribution(state, persona)
            content = _short_ai_contribution(str(response or ""))
            if not content:
                content = _short_ai_contribution(f"Building on {previous or state['topic']}, we should consider its practical effect.")
            _append_contribution(state, speaker, content)
            _save(db, record, state)
            return _public_state(state)

    @router.post("/api/gd/session/{public_id}/respond")
    async def gd_respond(public_id: str, request: Request, payload: dict, db: Session = Depends(get_db)):
        user = _auth_user(request, db, get_current_user)
        lock = _session_locks.setdefault(public_id, asyncio.Lock())
        async with lock:
            record = _get_record(db, public_id, user)
            state = _loads(record)
            if _expire_if_due(state):
                _save(db, record, state)
            if state["phase"] in {"completed", "expired"}:
                raise HTTPException(status_code=409, detail="This discussion has ended")
            if str(payload.get("turn_token", "")) != _public_state(state)["turn_token"]:
                raise HTTPException(status_code=409, detail="This response has already been submitted; refresh the discussion")
            speaker = _speaker(state)
            if not speaker or speaker["id"] != "user":
                raise HTTPException(status_code=409, detail="It is not your turn")
            content = str(payload.get("transcript", "")).strip()
            if not content:
                raise HTTPException(status_code=422, detail="Add or record your contribution before submitting")
            _append_contribution(state, speaker, content[:6000], {
                "input_mode": "voice" if payload.get("input_mode") == "voice" else "text",
                "duration_seconds": payload.get("duration_seconds"),
            })
            if state["phase"] == "completed":
                state["report"] = await _build_report(state, llm_service)
            _save(db, record, state)
            return _public_state(state)

    @router.post("/api/gd/transcribe")
    async def gd_transcribe(file: UploadFile = File(...), context_prompt: str = Form(default=""), request: Request = None, db: Session = Depends(get_db)):
        user = _auth_user(request, db, get_current_user)
        del user  # authentication only; this endpoint is GD-specific and creates no interview records
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio recording is empty")
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Recording is too large")
        if not llm_service.settings.groq_api_key:
            raise HTTPException(status_code=503, detail="Speech recognition is not configured on the server")
        suffix = Path(file.filename or "recording.webm").suffix.lower()
        if suffix not in {".webm", ".ogg", ".wav", ".mp3", ".m4a"}:
            suffix = ".webm"
        request_args = {
            "file": (f"gd-recording{suffix}", audio_bytes),
            "model": "whisper-large-v3-turbo",
            "language": "en", "response_format": "json", "temperature": 0.0,
        }
        if context_prompt.strip():
            request_args["prompt"] = context_prompt[:700]
        try:
            result = await asyncio.to_thread(transcribe_audio, request_args)
        except Exception as exc:
            logger.warning("GD transcription failed: type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Transcription is temporarily unavailable. Retry or enter your contribution by text.") from exc
        transcript = str(getattr(result, "text", "") or "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech could be transcribed from this recording")
        return {"transcript": transcript}

    return router
