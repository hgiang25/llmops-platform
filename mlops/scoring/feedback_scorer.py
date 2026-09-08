"""
Feedback Scorer — Processes user implicit/explicit feedback signals.

Converts raw feedback (thumbs_up, thumbs_down, regenerate) into a
structured signal for score aggregation.

IMPORTANT DESIGN PRINCIPLE:
    User feedback is a WEAK SUPERVISION SIGNAL, not absolute ground truth.
    - Thumbs down may indicate: wrong answer, wrong format, changed mind, etc.
    - Missing feedback ≠ positive feedback (feedback bias).
    - Only a fraction of users provide feedback (feedback coverage).

These limitations are documented and accounted for in the scoring pipeline.
"""

import logging

logger = logging.getLogger(__name__)


def compute_feedback_signal(
    thumbs_up: bool = False,
    thumbs_down: bool = False,
    regenerate: bool = False,
) -> dict:
    """
    Convert raw feedback into a structured signal.

    Args:
        thumbs_up: User clicked 👍.
        thumbs_down: User clicked 👎.
        regenerate: User requested response regeneration.

    Returns:
        {
            "feedback_signal": str ("positive" | "negative" | "strong_negative" | "absent"),
            "has_feedback": bool,
            "signal_strength": float (-1.0 to 1.0),
        }

    Signal interpretation:
        - "positive": User explicitly liked the response.
        - "negative": User explicitly disliked the response.
        - "strong_negative": User disliked AND requested regeneration.
        - "absent": No feedback provided (most common case).
    """
    # Handle contradictory feedback (both thumbs_up and thumbs_down)
    if thumbs_up and thumbs_down:
        logger.warning("Contradictory feedback: both thumbs_up and thumbs_down are True. "
                        "Treating as absent.")
        return {
            "feedback_signal": "absent",
            "has_feedback": False,
            "signal_strength": 0.0,
        }

    if thumbs_down and regenerate:
        return {
            "feedback_signal": "strong_negative",
            "has_feedback": True,
            "signal_strength": -1.0,
        }
    elif thumbs_down:
        return {
            "feedback_signal": "negative",
            "has_feedback": True,
            "signal_strength": -0.7,
        }
    elif regenerate:
        # Regenerate without explicit thumbs_down — moderate negative signal
        return {
            "feedback_signal": "negative",
            "has_feedback": True,
            "signal_strength": -0.5,
        }
    elif thumbs_up:
        return {
            "feedback_signal": "positive",
            "has_feedback": True,
            "signal_strength": 1.0,
        }
    else:
        # No feedback — the most common case.
        # IMPORTANT: This does NOT mean the response was good.
        return {
            "feedback_signal": "absent",
            "has_feedback": False,
            "signal_strength": 0.0,
        }
