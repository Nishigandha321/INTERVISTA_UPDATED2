"""
Reinforcement Learning Helpers (Simplified for Contextual Bandit)
==================================================================

Essential utilities for bandit-based adaptive learning:
- State discretization (convert continuous metrics to discrete states)
- Reward calculation (simple score improvement signal)

Features:
- No temporal difference (no next_state dependency)
- Running average of rewards per (state, action)
- Simple, interpretable reward signal
"""

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DISCRETIZATION
# =============================================================================

def get_state_id(avg_score: float, weak_topics_count: int) -> str:
    """
    Convert continuous user metrics to a discrete state.
    
    Args:
        avg_score: User's average score (0-100)
        weak_topics_count: Number of weak topics identified
    
    Returns:
        State ID in format "{score_level}-{weak_count}"
        Examples: "low-2", "medium-3", "high-0"
    
    Discretization:
        - avg_score < 50 → "low"
        - 50-75 → "medium"
        - > 75 → "high"
        - weak_count: exact count (0, 1, 2, 3+)
    """
    # Normalize score to 0-100 range
    avg_score = max(0, min(100, float(avg_score)))
    
    # Discretize score level
    if avg_score < 50:
        score_level = "low"
    elif avg_score <= 75:
        score_level = "medium"
    else:
        score_level = "high"
    
    # Cap weak topics count at 3+ 
    weak_count = min(int(weak_topics_count), 3)
    
    state_id = f"{score_level}-{weak_count}"
    
    logger.debug(
        f"State: score={avg_score:.1f} weak_topics={weak_topics_count} → state_id={state_id}"
    )
    
    return state_id


# =============================================================================
# REWARD CALCULATION (SIMPLE SCORE IMPROVEMENT)
# =============================================================================

def calculate_reward(
    current_score: float,
    previous_score: float,
    current_weak_topics: list | None = None,
    previous_weak_topics: list | None = None,
    current_confidence: float | None = None,
    previous_confidence: float | None = None,
) -> float:
    """
    Calculate an adaptive reward based on learning progression signals.

    Args:
        current_score: User's current score (0-100)
        previous_score: User's previous score (0-100)
        current_weak_topics: Current session weak topics
        previous_weak_topics: Previous session weak topics
        current_confidence: Current confidence score (0-100)
        previous_confidence: Previous confidence score (0-100)

    Returns:
        Reward normalized to [-1.0, +1.0]

    Components:
        0.5 × Score improvement
        0.3 × Weak-topic progress
        0.2 × Confidence improvement
    """
    curr = max(0.0, min(100.0, float(current_score)))
    prev = max(0.0, min(100.0, float(previous_score)))
    score_delta = (curr - prev) / 100.0

    current_weak_topics = [str(t).strip().lower() for t in (current_weak_topics or []) if str(t).strip()]
    previous_weak_topics = [str(t).strip().lower() for t in (previous_weak_topics or []) if str(t).strip()]

    prev_top = previous_weak_topics[:3]
    curr_top = current_weak_topics[:3]
    overlap = len(set(prev_top).intersection(set(curr_top)))
    weak_progress = max(0.0, min(1.0, (len(prev_top) - overlap) / 3.0))

    if current_confidence is None or previous_confidence is None:
        confidence_delta = 0.0
    else:
        curr_conf = max(0.0, min(100.0, float(current_confidence)))
        prev_conf = max(0.0, min(100.0, float(previous_confidence)))
        confidence_delta = (curr_conf - prev_conf) / 100.0

    reward = (
        0.5 * score_delta
        + 0.3 * weak_progress
        + 0.2 * confidence_delta
    )
    reward = max(-1.0, min(1.0, reward))

    # TODO: optionally incorporate course-resource completion ratio into reward
    # (e.g. fraction of CourseResource.marked_complete for the last course).

    logger.debug(
        f"Reward: current={curr:.1f}, previous={prev:.1f}, "
        f"score_delta={score_delta:+.3f}, weak_progress={weak_progress:+.3f}, "
        f"confidence_delta={confidence_delta:+.3f} → reward={reward:.3f}"
    )
    return reward

