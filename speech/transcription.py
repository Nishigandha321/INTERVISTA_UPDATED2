import os
import re
import time
import subprocess
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
    lower = cleaned.lower()
    for pattern, repl in _STT_REPLACEMENTS:
        lower = re.sub(pattern, repl, lower, flags=re.IGNORECASE)
    # Preserve sentence casing loosely: capitalize first char
    if lower:
        lower = lower[0].upper() + lower[1:] if len(lower) > 1 else lower.upper()
    return lower


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


def transcribe_audio(file_path: str) -> str:
    if whisper is None:
        print("[warning] whisper is not installed; audio transcription is unavailable.")
        return ""

    try:
        converted_path = f"converted_{int(time.time() * 1000)}.wav"
        _convert_audio(file_path, converted_path)
        model = _get_whisper_model()
        result = model.transcribe(
            converted_path,
            language="en",
            beam_size=5,
            temperature=0,
            initial_prompt="Technical job interview answer with complete sentences.",
        )
        return normalize_transcript(result.get("text", "").strip())
    except Exception as exc:
        print(f"[transcription] error: {exc}")
        return ""
