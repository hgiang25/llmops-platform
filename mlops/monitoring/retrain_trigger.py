"""
Retrain Trigger — Decision logic for when to retrain the Router model.

CORE PRINCIPLE:
    Data Drift alone does NOT trigger retraining.
    Routing Drift alone does NOT trigger retraining.
    Only QUALITY DEGRADATION with supporting evidence triggers retraining.

Retrain signal requires:
    1. Quality drop detected (statistical test)
    2. Routing failure evidence (elevated failure rate)
    3. Sufficient sample size (avoid false positives from small batches)

All thresholds are configurable via configs/monitoring.yaml.
"""

import logging
from typing import Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Drift Decision Matrix
DECISION_MATRIX = {
    (False, False, False): "healthy",
    (True, False, False): "input_distribution_changed",
    (False, True, False): "routing_distribution_changed",
    (True, True, False): "user_behavior_changed",
    (False, False, True): "response_quality_degradation",
    (True, False, True): "response_quality_degradation",
    (False, True, True): "routing_problem",
    (True, True, True): "potential_model_degradation",
}


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "monitoring.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_retrain_trigger(
    data_drift_detected: bool,
    routing_drift_detected: bool,
    quality_drop_detected: bool,
    quality_metrics: dict,
    config: Optional[dict] = None,
) -> dict:
    """
    Evaluate whether the Router model should be retrained.

    Args:
        data_drift_detected: Whether input data distribution has shifted.
        routing_drift_detected: Whether routing distribution has shifted.
        quality_drop_detected: Whether response quality has dropped.
        quality_metrics: Output from compute_quality_metrics().
        config: Monitoring configuration.

    Returns:
        {
            "retrain_recommended": bool,
            "retrain_reason": str or None,
            "overall_conclusion": str,
            "evidence": dict,
            "warnings": list[str],
        }
    """
    if config is None:
        config = load_config()

    retrain_cfg = config.get("retrain", {})
    quality_cfg = config.get("drift", {}).get("quality", {})

    require_quality_drop = retrain_cfg.get("require_quality_drop", True)
    require_routing_failure = retrain_cfg.get("require_routing_failure", True)
    require_minimum_samples = retrain_cfg.get("require_minimum_samples", True)
    minimum_samples = retrain_cfg.get("minimum_samples", 100)

    failure_threshold = quality_cfg.get("routing_failure_rate_threshold", 0.10)
    uncertain_warning = quality_cfg.get("uncertain_rate_warning", 0.30)

    # Extract metrics
    n_records = quality_metrics.get("n_records", 0)
    potential_failure_rate = quality_metrics.get("potential_failure_rate", 0.0)
    uncertain_rate = quality_metrics.get("uncertain_rate", 0.0)

    # Decision matrix lookup
    conclusion = DECISION_MATRIX.get(
        (data_drift_detected, routing_drift_detected, quality_drop_detected),
        "unknown",
    )

    # Retrain logic
    retrain_recommended = False
    retrain_reason = None
    warnings = []

    # Check 1: Quality drop
    has_quality_drop = quality_drop_detected if require_quality_drop else True

    # Check 2: Routing failure evidence
    has_routing_failure = (
        potential_failure_rate > failure_threshold
    ) if require_routing_failure else True

    # Check 3: Sufficient samples
    has_enough_samples = (
        n_records >= minimum_samples
    ) if require_minimum_samples else True

    # All conditions must be met
    if has_quality_drop and has_routing_failure and has_enough_samples:
        retrain_recommended = True
        retrain_reason = (
            f"quality_drop={quality_drop_detected}, "
            f"failure_rate={potential_failure_rate:.2%} > {failure_threshold:.2%}, "
            f"samples={n_records} >= {minimum_samples}"
        )
    elif has_quality_drop and not has_enough_samples:
        warnings.append(
            f"Quality drop detected but insufficient samples ({n_records} < {minimum_samples}). "
            "Waiting for more data before recommending retraining."
        )
    elif has_quality_drop and not has_routing_failure:
        warnings.append(
            f"Quality drop detected but routing failure rate ({potential_failure_rate:.2%}) "
            f"is below threshold ({failure_threshold:.2%}). "
            "Quality issue may not be routing-related."
        )

    # Warning for high uncertainty
    if uncertain_rate > uncertain_warning:
        warnings.append(
            f"High uncertainty rate ({uncertain_rate:.2%} > {uncertain_warning:.2%}). "
            "Many queries have insufficient evidence for assessment. "
            "Consider building a larger/more diverse reference dataset."
        )

    # Warning if data drift but no quality drop
    if data_drift_detected and not quality_drop_detected:
        warnings.append(
            "Data distribution has shifted but quality remains stable. "
            "This likely reflects changing user behavior, not model degradation."
        )

    return {
        "retrain_recommended": retrain_recommended,
        "retrain_reason": retrain_reason,
        "overall_conclusion": conclusion,
        "evidence": {
            "data_drift_detected": data_drift_detected,
            "routing_drift_detected": routing_drift_detected,
            "quality_drop_detected": quality_drop_detected,
            "potential_failure_rate": potential_failure_rate,
            "uncertain_rate": uncertain_rate,
            "n_records": n_records,
            "minimum_samples_required": minimum_samples,
        },
        "warnings": warnings,
    }
