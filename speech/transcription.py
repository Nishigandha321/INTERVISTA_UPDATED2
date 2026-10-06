import os
import re
import subprocess
import sys
import tempfile
import types
from typing import Optional

# Common STT mis-hearings — normalize before scoring (not a full spell-checker)
_STT_REPLACEMENTS = (
    (r"\bteh\b", "the"),
    (r"\bwich\b", "which"),
    (r"\bthier\b", "their"),
    (r"\brecieve\b", "receive"),
    (r"\bseperate\b", "separate"),
    (r"\bdefinately\b", "definitely"),
    (r"\bimplemention\b", "implementation"),
    (r"\balot\b", "a lot"),
    (r"\bcuz\b", "because"),
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
)


def normalize_transcript(text: str) -> str:
    """
    Light cleanup for speech-to-text answers before LLM/heuristic evaluation.
    Reduces penalty from minor transcription noise.
    """
    if not text:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(.)\1{3,}", r"\1\1", cleaned)  # collapse long char repeats
    for pattern, repl in _STT_REPLACEMENTS:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    return cleaned


_TECH_CONTEXT = re.compile(
    r"\b(technical|technolog(?:y|ies)|stack|framework|library|database|backend|project|"
    r"programming|language|api|server|deployment|configuration)\b", re.IGNORECASE
)
_TECH_ALIASES = (
    (r"\b(?:dot\s+e\s*n\s*v|d[\s.-]*o[\s.-]*t\s+e[\s.-]*n[\s.-]*v|dot\s+env)\b", ".env"),
    (r"\b(?:c|see)\s+plus\s+plus\b", "C++"),
    (r"\bnode\s+(?:dot\s*)?js\b", "Node.js"),
    (r"\bpost\s+gres(?:ql)?\b", "PostgreSQL"),
)
_KNOWN_TECH_TERMS = {
    "python", "c++", "java", "javascript", "typescript", "fastapi", "postgresql",
    "postgres", "sqlalchemy", "jwt", "api", "rest", "react", "node.js", "docker",
    "git", "github", "kubernetes", "tensorflow", "opencv", "cnn", "rnn", "llm",
    "rag", "nlp", "ml", "ai", ".env", "json", "xml", "http", "https", "oauth",
    "crud", "orm", "sql", "nosql", "django", "flask", "redis", "mongodb", "aws",
    "azure", "gcp", "kafka", "graphql", "sqlite", "mysql", "pytorch",
}


def normalize_candidate_transcript(
    raw_transcript: str,
    interview_type: str = "technical",
    context: str = "",
) -> dict:
    """Conservatively normalize obvious technical ASR formatting while retaining raw text."""
    raw = str(raw_transcript or "")
    normalized = " ".join(raw.split())
    changed = False
    technical = str(interview_type or "").casefold() == "technical"
    if technical and _TECH_CONTEXT.search(context or ""):
        for pattern, replacement in _TECH_ALIASES:
            updated = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            changed = changed or updated != normalized
            normalized = updated

    suspicious_term = ""
    # An isolated, unfamiliar name in a technology-stack answer is ambiguous. Ask
    # the candidate instead of guessing a correction or treating it as a technology.
    if technical and _TECH_CONTEXT.search(context or ""):
        terms = re.findall(r"\b(?:use(?:d)?|using|with|including)\s+([A-Z][a-z]{3,})\b", raw)
        known = {term.casefold() for term in _KNOWN_TECH_TERMS}
        for term in terms:
            if term.casefold() not in known and not re.search(r"\b" + re.escape(term) + r"\b", context, re.IGNORECASE):
                suspicious_term = term
                break
    return {
        "raw_transcript": raw,
        "normalized_transcript": normalized,
        "normalization_applied": changed,
        "suspicious_term": suspicious_term,
    }


FILLER_WORDS = {
    "um",
    "uh",
    "uhm",
    "hmm",
    "like",
    "so",
    "well",
    "you know",
    "actually",
    "basically",
    "literally",
    "totally",
    "okay",
    "right",
    "kind of",
    "sort of",
}

try:
    import numba  # noqa: F401
except (ImportError, OSError):
    # Whisper only uses numba to JIT its optional word-timing helpers. Those
    # helpers are not enabled for interview transcription, so keep the Python
    # implementations usable when the optional LLVM runtime is unavailable.
    numba_fallback = types.ModuleType("numba")

    def _no_op_jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    numba_fallback.jit = _no_op_jit
    sys.modules["numba"] = numba_fallback

try:
    import whisper
except ImportError:
    whisper = None

_whisper_model = None


def _ensure_temp_path(temp_path: str):
    directory = os.path.dirname(temp_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _convert_audio(input_path: str, output_path: str):
    _ensure_temp_path(output_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        output_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def analyze_speech_delivery(answer: str, duration_seconds: float) -> dict:
    answer = normalize_transcript(answer)
    if not answer.strip():
        return {
            "word_count": 0,
            "duration_seconds": duration_seconds,
            "speaking_pace_wpm": 0,
            "filler_word_count": 0,
            "filler_words_found": [],
            "clarity_score": 0,
            "engagement_score": 0,
        }

    words = answer.split()
    word_count = len(words)
    duration_minutes = max(duration_seconds / 60.0, 0.01)
    wpm = word_count / duration_minutes
    cleaned = [w.lower().strip(".,!?") for w in words]
    filler_pattern = r"\b(?:" + "|".join(
        re.escape(filler).replace(r"\ ", r"\s+")
        for filler in sorted(FILLER_WORDS, key=len, reverse=True)
    ) + r")\b"
    fillers = [match.group(0).lower() for match in re.finditer(filler_pattern, answer, re.IGNORECASE)]
    filler_rate = len(fillers) / max(word_count, 1)
    sentences = re.split(r"[.!?]+", answer)
    sentences = [s for s in sentences if s.strip()]
    avg_sentence = word_count / max(len(sentences), 1)
    clarity = 40
    clarity -= filler_rate * 150
    if avg_sentence > 30:
        clarity -= avg_sentence - 30
    clarity = max(0, min(100, round(clarity)))
    unique_words = len(set(cleaned))
    vocab_ratio = unique_words / max(word_count, 1)
    engagement = min(100, round(vocab_ratio * 100))

    return {
        "word_count": word_count,
        "duration_seconds": round(duration_seconds, 1),
        "speaking_pace_wpm": round(wpm, 1),
        "filler_word_count": len(fillers),
        "filler_words_found": fillers,
        "clarity_score": clarity,
        "engagement_score": engagement,
    }


def compute_confidence_score(speech_analyses: list) -> int:
    if not speech_analyses:
        return 0

    total_fillers = sum(a.get("filler_word_count", 0) for a in speech_analyses)
    total_words = sum(a.get("word_count", 0) for a in speech_analyses)
    filler_rate = total_fillers / max(total_words, 1)
    filler_conf = max(0, 100 - filler_rate * 300)

    paces = [a.get("speaking_pace_wpm", 0) for a in speech_analyses]
    avg_pace = sum(paces) / len(paces) if paces else 0
    pace_conf = 100 if 120 <= avg_pace <= 160 else 70

    avg_words = total_words / len(speech_analyses) if speech_analyses else 0
    length_conf = 90 if avg_words > 60 else 60

    score = 0.4 * filler_conf + 0.3 * pace_conf + 0.3 * length_conf
    return round(max(0, min(100, score)))


def compute_overall_score(content_avg, clarity_avg, engagement_avg, answers=None):
    attempted = [a for a in (answers or []) if a.get("answer") not in ["(skipped)", "(no response)", ""]]
    if len(attempted) == 0:
        return 0

    score = 0.5 * content_avg + 0.3 * clarity_avg + 0.2 * engagement_avg
    return round(max(0, min(100, score)), 1)


def compute_recruiter_verdict(overall_score: float, role: str):
    if overall_score >= 70:
        rec = "SHORTLISTED"
    elif overall_score >= 50:
        rec = "BORDERLINE"
    else:
        rec = "REJECT"

    return {
        "recommendation": rec,
        "confidence_level": "High" if overall_score >= 70 else "Medium",
        "suitable_roles": [role],
    }


def _get_whisper_model():
    global _whisper_model
    if whisper is None:
        raise ImportError("whisper module is not installed")
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")
    return _whisper_model


def transcribe_audio(file_path: str, context_prompt: str = "") -> str:
    if whisper is None:
        raise RuntimeError("Whisper is not installed")

    converted_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            converted_path = temp_audio.name
        _convert_audio(file_path, converted_path)
        model = _get_whisper_model()
        result = model.transcribe(
            converted_path,
            language="en",
            beam_size=5,
            temperature=0,
            initial_prompt=(context_prompt or "Interview response with complete sentences.")[:500],
        )
        return normalize_transcript(result.get("text", "").strip())
    except Exception as exc:
        print(f"[transcription] error: {exc}")
        raise RuntimeError("Audio transcription failed") from exc
    finally:
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except OSError:
                pass
