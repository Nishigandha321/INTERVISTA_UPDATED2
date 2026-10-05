"""Lightweight, deterministic turn analysis and interview action selection."""

import re
from difflib import SequenceMatcher

ACTION_SET = {
    "FOLLOW_UP", "CLARIFICATION", "DEEPEN", "CHALLENGE",
    "CHANGE_TOPIC", "BEHAVIORAL_PROBE", "MOVE_ON", "END_INTERVIEW",
}
QUESTION_CATEGORIES = {
    "resume-based", "jd-based", "missing-skill", "behavioral-company-fit", "behavioral",
    "situational", "technical", "logical", "project-specific", "coding", "company-specific",
    "skill-based", "project-based",
}

_TECH_TERMS = (
    "FastAPI", "Django", "Flask", "PostgreSQL", "Postgres", "MySQL", "SQLite",
    "MongoDB", "Redis", "JWT", "OAuth", "REST", "GraphQL", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "React", "Node.js", "Python", "Java", "C++", "TypeScript",
    "JavaScript", "async", "asynchronous", "SQL", "indexing", "normalization", "cache",
    "caching", "queue", "Kafka", "microservices", "API", "authentication", "authorization",
    "transaction", "ACID", "replication", "sharding", "load balancing", "concurrency",
    "thread", "lock", "event loop", "B-tree", "hash map", "binary search", "DFS", "BFS",
    "TCP", "HTTP", "TLS", "OAuth2", "pytest", "CI/CD", "Git", "Linux", "SQLAlchemy",
    "Spring Boot", "health checks", "health check", "error handling", "retries", "retry",
    "JSON", "serialization", "deserialization", "input validation", "null checks", "null",
    "scalability", "scalable architecture", "system design", "observability", "logging",
)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with",
    "by", "from", "as", "is", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "i", "we", "they", "he", "she", "you", "my", "our", "their", "me",
    "our", "so", "because", "then", "if", "when", "which", "what", "how", "why", "use",
    "used", "using", "have", "has", "had", "do", "does", "did", "can", "could", "would",
    "customer", "customers", "user", "users", "client", "clients", "people", "team",
    "company", "applied", "apply", "applying", "approach", "thing", "things",
}
_GENERIC_FOCUS = "your approach"


def _normalize(text: str) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9+#.]+", (text or "").casefold()))
    for source, replacement in (
        ("what made you", "why did you"),
        ("select", "choose"),
        ("selected", "choose"),
        ("choosing", "choose"),
        ("selection", "choose"),
    ):
        normalized = normalized.replace(source, replacement)
    return normalized


def question_is_duplicate(candidate: str, previous: list[str]) -> bool:
    """Detect exact and close paraphrase duplicates without an embedding call."""
    normalized = _normalize(candidate)
    if not normalized:
        return True
    candidate_tokens = set(normalized.split())
    for old in previous:
        old_normalized = _normalize(old)
        if normalized == old_normalized or SequenceMatcher(None, normalized, old_normalized).ratio() >= 0.84:
            return True
        old_tokens = set(old_normalized.split())
        if candidate_tokens and old_tokens:
            overlap = len(candidate_tokens & old_tokens) / len(candidate_tokens | old_tokens)
            if overlap >= 0.78:
                return True
    return False


def validate_generated_question(question: str, category: str, previous: list[str], action: str,
                                focus_keyword: str = "", current_category: str = "",
                                excluded_topics: list[str] | None = None) -> tuple[str, str]:
    """Enforce the generated-question contract before changing interview state."""
    question = " ".join((question or "").split())
    category = (category or "").strip()
    if not question or len(question.split()) > 45 or question.count("?") != 1:
        raise ValueError("Question failed length or format validation")
    if re.search(r"\b(?:apply|applied|applying)\s+(?:the\s+)?(?:customers?|users?|clients?)\b", question, re.I):
        raise ValueError("Question treats an audience noun as a design choice")
    known_design_terms = {term.casefold() for term in _TECH_TERMS}
    if (
        action == "CHALLENGE"
        and focus_keyword.casefold() not in known_design_terms
        and re.search(r"\b(drawbacks?|trade[- ]?offs?|limitations?|failure cases?)\b", question, re.I)
    ):
        raise ValueError("Question invents a trade-off for an unnamed design choice")
    if category not in QUESTION_CATEGORIES:
        raise ValueError("Question response contains an invalid category")
    if question_is_duplicate(question, previous):
        raise ValueError("Question duplicates an earlier question")
    for topic in excluded_topics or []:
        topic = str(topic or "").strip()
        if topic and re.search(r"(?<!\w)" + re.escape(topic) + r"(?!\w)", question, re.I):
            raise ValueError("Question targets a candidate-declared knowledge gap")
    probe_actions = {"FOLLOW_UP", "CLARIFICATION", "DEEPEN", "CHALLENGE", "BEHAVIORAL_PROBE"}
    if action in probe_actions and focus_keyword.casefold() != _GENERIC_FOCUS and focus_keyword.casefold() not in question.casefold():
        raise ValueError("Follow-up question is not grounded in the candidate's answer")
    if action in {"CHANGE_TOPIC", "MOVE_ON"} and category == current_category:
        raise ValueError("The new main question must use a different category")
    return question, category


def validate_candidate_answer(answer: str, skip: bool = False, max_characters: int = 12000) -> str:
    """Reject empty speech transcripts while preserving the explicit skip control."""
    answer = str(answer or "").strip()
    if not answer and not skip:
        raise ValueError("Please say an answer before continuing")
    if len(answer) > max_characters:
        raise ValueError("Answer is too long")
    return answer or "(skipped)"


def analyze_answer(question: str, answer: str, topic: str = "") -> dict:
    """Produce inexpensive structured signals for deciding the next interviewer action."""
    clean_answer = " ".join((answer or "").split())
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", clean_answer)
    lower = clean_answer.casefold()
    concepts = []
    for term in _TECH_TERMS:
        if re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", lower):
            concepts.append(term)
    # Capture project names and other named technologies when speech text has no known term.
    for name in re.findall(r"\b[A-Z][A-Za-z0-9+#.-]{2,}\b", clean_answer):
        if name.casefold() not in {value.casefold() for value in concepts} and name.casefold() not in _STOPWORDS:
            concepts.append(name)
    clauses = [part.strip() for part in re.split(r"[,;.!?]+", clean_answer) if len(part.split()) >= 3]
    evidence_markers = (
        "i built", "i implemented", "i designed", "i chose", "i measured", "because",
        "for example", "we measured", "measured", "reduced", "improved", "result",
        "trade-off", "tradeoff", "to prevent", "to ensure", "after profiling",
    )
    evidence = any(marker in lower for marker in evidence_markers)
    questionable = []
    if "jwt" in lower and re.search(r"\b(encrypted|encryption|secret payload)\b", lower):
        questionable.append("JWT encoding/signing may be confused with payload encryption")
    if re.search(r"password.{0,30}(plaintext|plain text|clear text)", lower):
        questionable.append("Password storage appears to omit hashing")
    if re.search(r"(sql|query).{0,40}(string concatenation|concatenat(e|ing)).{0,30}(safe|secure)", lower):
        questionable.append("SQL string concatenation is described as safe")
    if re.search(r"\b(async|asynchronous)\b.{0,35}\b(always faster|always faster than|guarantees speed)\b", lower):
        questionable.append("Asynchronous execution is claimed to guarantee faster performance")
    question_terms = {
        term for term in re.findall(r"[a-z][a-z0-9+#.-]+", (question or "").casefold())
        if term not in _STOPWORDS and term not in {"tell", "explain", "describe", "discuss", "walk", "through", "share", "give", "example", "project"}
    }
    question_concepts = [term for term in _TECH_TERMS if re.search(
        r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", (question or "").casefold()
    )]
    answer_terms = {term.casefold() for term in words if term.casefold() not in _STOPWORDS}
    overlap_count = len(question_terms & answer_terms)
    gap_match = re.search(
        r"\b(?:i\s+(?:do not|don't|dont|cannot|can't)\s+know|"
        r"i\s+(?:have|got)\s+no\s+experience\s+with|"
        r"i(?:'m| am)\s+not\s+familiar\s+with)\b", lower
    )
    knowledge_gap_statement = gap_match is not None
    low_confidence = bool(re.search(
        r"\b(?:not\s+sure|unsure|not\s+confident|i\s+(?:think|guess)|maybe)\b", lower
    ))
    candidate_stuck = knowledge_gap_statement or bool(
        re.match(r"^(?:i (?:do not|don't|cannot|can't) know|not sure|no idea)\b", lower)
    )
    knowledge_gap_topics = []
    if knowledge_gap_statement:
        # Prefer skills the candidate named; use the question's skill when they
        # only said "I don't know it". Never turn the category label into a skill.
        gap_tail = re.split(
            r"\b(?:but|however|although|while)\b|[,;.!?]",
            lower[gap_match.end():], maxsplit=1,
        )[0]
        for term in _TECH_TERMS:
            if re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", gap_tail):
                knowledge_gap_topics.append(term)
        if not knowledge_gap_topics:
            for term in _TECH_TERMS:
                if re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", (question or "").casefold()):
                    knowledge_gap_topics.append(term)
        if not knowledge_gap_topics and topic and topic.casefold() not in {"technical", "hr", "behavioral"}:
            knowledge_gap_topics.append(topic)
    knowledge_gap_topics = list(dict.fromkeys(knowledge_gap_topics))
    explicit_knowledge_gap = bool(knowledge_gap_topics)
    non_response = lower in {"skip", "(skipped)"} or candidate_stuck
    answered = len(words) >= 4 and not non_response and (overlap_count > 0 or len(words) >= 7)
    relevance = round(min(1.0, 0.45 + overlap_count * 0.15), 2) if answered else 0.1
    quality = min(1.0, (min(len(words), 60) / 60) * 0.45 + relevance * 0.15 + (0.2 if concepts else 0) + (0.15 if evidence else 0) + (0.05 if clauses else 0))
    missing = []
    if not concepts:
        missing.append("specific technical concept or design choice")
    if not evidence:
        missing.append("supporting example, rationale, or outcome")
    return {
        "topic": topic or (concepts[0] if concepts else "General interview skills"),
        "skill": topic or "Communication and problem solving",
        "answer_quality": round(quality, 2),
        "key_points": clauses[:4],
        "concepts_mentioned": concepts[:8],
        "question_concepts": question_concepts,
        "evidence_provided": evidence,
        "missing_points": missing,
        "weaknesses": ["Answer is brief or lacks a concrete example"] if quality < 0.35 else [],
        "contradiction_signals": bool(re.search(r"\b(always|never)\b", lower) and re.search(r"\b(but|however|although)\b", lower)),
        "technically_questionable": questionable,
        "depth": "strong" if quality >= 0.72 else "developing" if quality >= 0.35 else "limited",
        "relevance": relevance,
        "relevance_score": round(relevance * 100),
        "answered_question": answered,
        "candidate_stuck": candidate_stuck,
        "explicit_knowledge_gap": explicit_knowledge_gap,
        "knowledge_gap_topics": knowledge_gap_topics,
        "low_confidence": low_confidence,
        "probe_worthy": bool(concepts or evidence or len(words) >= 12),
    }


def select_action(analysis: dict, probe_count: int, max_probes: int, turn_count: int,
                  max_turns: int, main_question_count: int, max_main_questions: int,
                  current_category: str = "technical") -> str:
    """Select an action while applying interview limits before any generation call."""
    if turn_count >= max_turns:
        return "END_INTERVIEW"
    if analysis.get("explicit_knowledge_gap"):
        return "CHANGE_TOPIC" if main_question_count < max_main_questions else "END_INTERVIEW"
    if probe_count >= max_probes:
        return "CHANGE_TOPIC" if main_question_count < max_main_questions else "END_INTERVIEW"
    if not analysis["answered_question"] and not analysis["probe_worthy"]:
        if main_question_count >= max_main_questions:
            return "END_INTERVIEW"
        if analysis.get("candidate_stuck"):
            return "CLARIFICATION"
        return "MOVE_ON" if analysis.get("depth") == "limited" else "CLARIFICATION"
    if not analysis["answered_question"]:
        return "CLARIFICATION" if probe_count < max_probes else "MOVE_ON"
    if analysis["contradiction_signals"] or analysis.get("technically_questionable"):
        return "CLARIFICATION"
    category = (current_category or "").casefold()
    if any(value in category for value in ("behavioral", "hr", "situational")) and not analysis["evidence_provided"]:
        return "BEHAVIORAL_PROBE"
    if analysis["answer_quality"] >= 0.72 and probe_count == 0:
        return "CHALLENGE"
    if analysis["probe_worthy"]:
        return "DEEPEN" if analysis["answer_quality"] >= 0.35 else "FOLLOW_UP"
    return "CLARIFICATION"


def choose_focus_keyword(answer: str, analysis: dict) -> str:
    """Pick an answer-grounded phrase for follow-up validation and fallback wording."""
    concepts = analysis.get("concepts_mentioned") or []
    answer_lower = (answer or "").casefold()
    known_concepts = [term for term in _TECH_TERMS if re.search(
        r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", answer_lower
    )]
    if concepts:
        generic = {"api", "sql", "authentication", "authorization", "async", "asynchronous", "cache", "caching", "thread", "transaction", "concurrency", "indexing", "normalization"}
        concrete = [term for term in known_concepts if str(term).casefold() not in generic]
        if concrete:
            return str(concrete[-1])
        return str((known_concepts or concepts)[-1])
    candidates = [word.strip(".,!?;:()[]{}\"'") for word in (answer or "").split()]
    candidates = [word for word in candidates if len(word) > 3 and word.casefold() not in _STOPWORDS]
    return candidates[-1] if candidates else _GENERIC_FOCUS


def fallback_question(action: str, keyword: str, question_history: list[str]) -> str:
    """Create a concise deterministic question anchored to the candidate's own wording."""
    if keyword.casefold() == _GENERIC_FOCUS:
        return "Could you clarify the specific choice or action you meant in your answer?"
    if keyword.casefold() not in {term.casefold() for term in _TECH_TERMS}:
        return f"Could you clarify what you meant by ‘{keyword}’ in that answer?"
    prompts = {
        "CLARIFICATION": f"Could you clarify how you used {keyword} in that approach?",
        "CHALLENGE": f"What failure case did you consider for {keyword}?",
        "BEHAVIORAL_PROBE": f"What specific action did you take with {keyword}, and what changed afterward?",
        "FOLLOW_UP": f"How did {keyword} influence the outcome of your approach?",
        "DEEPEN": f"How did you apply {keyword}, and what trade-off did that involve?",
    }
    candidate = prompts.get(action, f"Could you explain your approach to {keyword} in more detail?")
    if not question_is_duplicate(candidate, question_history):
        return candidate
    alternatives = [
        f"What led you to choose {keyword} for this situation?",
        f"How would {keyword} behave if the system had to handle twice the workload?",
        f"What would you change about your use of {keyword} now?",
    ]
    return next((item for item in alternatives if not question_is_duplicate(item, question_history)), candidate)
