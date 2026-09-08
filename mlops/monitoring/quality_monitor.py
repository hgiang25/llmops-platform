"""
Quality Monitor — Response quality metrics computation.

Computes quality metrics from scored production data for drift detection
and monitoring dashboards.

IMPORTANT PRINCIPLES:
    - missing_feedback ≠ positive_feedback
    - Always distinguish:
        * negative_feedback / all_queries
        * negative_feedback / feedback_bearing_queries
    - Feedback has coverage bias: dissatisfied users are more likely to vote.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_quality_metrics(scored_records: list[dict]) -> dict:
    """
    Compute quality metrics from scored production records.

    Args:
        scored_records: List of records output from BatchScorer.
            Expected fields: final_quality_score, feedback_signal,
            has_feedback, routing_assessment, is_uncertain, regenerate.

    Returns:
        Dict with quality metrics.
    """
    if not scored_records:
        return {"error": "No records to analyze", "n_records": 0}

    n = len(scored_records)

    # --- Quality Scores ---
    quality_scores = [
        r["final_quality_score"]
        for r in scored_records
        if r.get("final_quality_score") is not None
    ]

    mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    median_quality = _median(quality_scores)

    # --- Feedback Metrics ---
    feedback_records = [r for r in scored_records if r.get("has_feedback", False)]
    feedback_count = len(feedback_records)
    feedback_coverage = feedback_count / n

    # Negative feedback
    negative_records = [
        r for r in scored_records
        if r.get("feedback_signal") in ("negative", "strong_negative")
    ]
    negative_count = len(negative_records)

    # IMPORTANT: Two different rates — document both clearly
    # Rate among ALL queries (including those without feedback)
    negative_rate_overall = negative_count / n
    # Rate among FEEDBACK-BEARING queries only
    negative_rate_feedback = negative_count / feedback_count if feedback_count > 0 else 0.0

    # Regenerate
    regenerate_records = [r for r in scored_records if r.get("regenerate", False)]
    regenerate_count = len(regenerate_records)
    regenerate_rate = regenerate_count / n

    # Positive feedback
    positive_records = [
        r for r in scored_records
        if r.get("feedback_signal") == "positive"
    ]
    positive_count = len(positive_records)
    positive_rate_overall = positive_count / n

    # --- Routing Assessment Metrics ---
    assessment_counts = {}
    for r in scored_records:
        assessment = r.get("routing_assessment", "unknown")
        assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1

    potential_failure_count = assessment_counts.get("potential_failure", 0)
    potential_failure_rate = potential_failure_count / n

    uncertain_count = (
        assessment_counts.get("unknown", 0)
        + assessment_counts.get("insufficient_evidence", 0)
    )
    uncertain_rate = uncertain_count / n

    correct_count = assessment_counts.get("correct", 0)
    correct_rate = correct_count / n

    over_routed_count = assessment_counts.get("over_routed", 0)
    over_routed_rate = over_routed_count / n

    # --- KNN Confidence ---
    confidences = [
        r.get("knn_confidence", 0.0)
        for r in scored_records
    ]
    mean_knn_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "n_records": n,
        # Quality scores
        "mean_quality": round(mean_quality, 4),
        "median_quality": round(median_quality, 4),
        # Feedback coverage
        "feedback_count": feedback_count,
        "feedback_coverage": round(feedback_coverage, 4),
        # Negative feedback — BOTH rates documented
        "negative_feedback_count": negative_count,
        "negative_feedback_rate_overall": round(negative_rate_overall, 4),
        "negative_feedback_rate_of_feedback": round(negative_rate_feedback, 4),
        # Positive feedback
        "positive_feedback_count": positive_count,
        "positive_feedback_rate_overall": round(positive_rate_overall, 4),
        # Regenerate
        "regenerate_count": regenerate_count,
        "regenerate_rate": round(regenerate_rate, 4),
        # Routing assessments
        "routing_assessment_distribution": assessment_counts,
        "potential_failure_count": potential_failure_count,
        "potential_failure_rate": round(potential_failure_rate, 4),
        "correct_rate": round(correct_rate, 4),
        "over_routed_rate": round(over_routed_rate, 4),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 4),
        # KNN confidence
        "mean_knn_confidence": round(mean_knn_confidence, 4),
    }


def _median(values: list[float]) -> float:
    """Compute median of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]
