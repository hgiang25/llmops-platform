"""
Score Aggregator — Combines Semantic KNN and User Feedback into final assessment.

This module implements the configurable policy for merging two weak signals:
1. Semantic KNN: base assessment from reference dataset similarity
2. User Feedback: implicit/explicit human behavioral signal

The output is an estimated response quality score and routing assessment.

TERMINOLOGY (enforced throughout):
    - "estimated_quality" — NOT "ground_truth_quality"
    - "routing_assessment" — NOT "routing_correctness"
    - "quality_signal" — NOT "quality_label"
    - "pseudo_label" — when used for downstream training candidate selection
"""

import logging
from typing import Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Valid routing assessment values
ROUTING_ASSESSMENTS = {
    "correct",                # KNN agrees with router + positive/absent feedback
    "potential_failure",      # Evidence suggests router may have been wrong
    "over_routed",            # Router used stronger model than needed
    "unknown",                # Insufficient evidence to assess
    "insufficient_evidence",  # KNN uncertain + no feedback
}


def load_monitoring_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "monitoring.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def aggregate_scores(
    knn_result: dict,
    feedback_result: dict,
    router_prediction: Optional[int] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Combine KNN score and user feedback into final quality assessment.

    Args:
        knn_result: Output from SemanticKNN.score_query().
        feedback_result: Output from compute_feedback_signal().
        router_prediction: The router model's predicted label (0/1/2). Optional.
        config: Monitoring configuration. Loaded from file if None.

    Returns:
        {
            "final_quality_score": float (1-5 scale),
            "quality_confidence": float (0-1),
            "routing_assessment": str,
            "scoring_method": str,
            "is_uncertain": bool,
            ...KNN fields,
            ...feedback fields,
        }
    """
    if config is None:
        config = load_monitoring_config()

    feedback_cfg = config.get("feedback", {})
    thumbs_up_boost = feedback_cfg.get("thumbs_up_boost", 1.5)
    thumbs_down_cap = feedback_cfg.get("thumbs_down_cap", 2.5)
    regenerate_cap = feedback_cfg.get("regenerate_cap", 2.0)

    # Base score from KNN
    base_score = knn_result.get("knn_quality_estimate", 3.0)
    knn_confidence = knn_result.get("knn_confidence", 0.0)
    knn_label = knn_result.get("knn_label")
    is_uncertain = knn_result.get("is_uncertain", True)

    # Feedback signal
    feedback_signal = feedback_result.get("feedback_signal", "absent")
    has_feedback = feedback_result.get("has_feedback", False)

    # --- Score adjustment based on feedback ---
    final_score = base_score

    if feedback_signal == "positive":
        # User liked it — boost score (but don't exceed 5)
        final_score = max(base_score, thumbs_up_boost * 3.0)  # At least 4.5
        final_score = min(final_score, 5.0)
    elif feedback_signal == "strong_negative":
        # User disliked + regenerated — strong negative
        final_score = min(base_score, regenerate_cap)
    elif feedback_signal == "negative":
        # User disliked — moderate negative
        final_score = min(base_score, thumbs_down_cap)
    # If "absent" — keep base_score from KNN

    # --- Confidence ---
    # Confidence is higher when we have both KNN and feedback signals
    if has_feedback and not is_uncertain:
        quality_confidence = min(knn_confidence + 0.15, 1.0)
        scoring_method = "semantic_knn+user_feedback"
    elif has_feedback and is_uncertain:
        quality_confidence = 0.4  # Feedback provides some signal even without KNN
        scoring_method = "user_feedback_only"
    elif not has_feedback and not is_uncertain:
        quality_confidence = knn_confidence
        scoring_method = "semantic_knn_only"
    else:
        quality_confidence = 0.1  # No KNN, no feedback
        scoring_method = "no_signal"

    # --- Routing Assessment ---
    routing_assessment = _assess_routing(
        knn_label=knn_label,
        router_prediction=router_prediction,
        feedback_signal=feedback_signal,
        is_uncertain=is_uncertain,
    )

    return {
        # Final scores
        "final_quality_score": round(final_score, 2),
        "quality_confidence": round(quality_confidence, 4),
        "routing_assessment": routing_assessment,
        "scoring_method": scoring_method,
        "is_uncertain": is_uncertain and not has_feedback,
        # KNN details (pass through)
        "knn_label": knn_label,
        "knn_label_name": knn_result.get("knn_label_name", "unknown"),
        "knn_confidence": knn_confidence,
        "knn_quality_estimate": knn_result.get("knn_quality_estimate", 3.0),
        # Feedback details (pass through)
        "feedback_signal": feedback_signal,
        "has_feedback": has_feedback,
    }


def _assess_routing(
    knn_label: Optional[int],
    router_prediction: Optional[int],
    feedback_signal: str,
    is_uncertain: bool,
) -> str:
    """
    Assess whether the router's routing decision was appropriate.

    This is NOT a definitive correctness judgment — it is an estimated
    assessment based on available evidence.

    Logic:
        1. If KNN is uncertain and no feedback → "insufficient_evidence"
        2. If feedback is negative → "potential_failure"
        3. If KNN agrees with router → "correct" (estimated)
        4. If KNN says weaker model would suffice → "over_routed"
        5. If KNN says stronger model needed → "potential_failure"
    """
    # Case 1: No evidence
    if is_uncertain and feedback_signal == "absent":
        return "insufficient_evidence"

    # Case 2: Negative feedback overrides KNN
    if feedback_signal in ("negative", "strong_negative"):
        return "potential_failure"

    # Case 3-5: Compare KNN label with router prediction (if available)
    if router_prediction is not None and knn_label is not None:
        if knn_label == router_prediction:
            return "correct"
        elif knn_label < router_prediction:
            # KNN says weaker model would suffice
            return "over_routed"
        else:
            # KNN says stronger model needed
            return "potential_failure"

    # Case 6: KNN uncertain but positive feedback
    if feedback_signal == "positive":
        return "correct"

    return "unknown"
