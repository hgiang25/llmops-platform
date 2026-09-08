"""
Test the 4 synthetic scenarios for the Monitoring & Retrain Trigger matrix.

Scenarios:
1. Healthy: No drift, no quality drop.
2. User Behavior Change: Data drift detected, but quality remains stable.
3. Routing Problem: No data drift, but routing failures spike.
4. Model Degradation: Data drift + Routing failures + Quality drop.
"""

# pyrefly: ignore [missing-import]
import pytest
from mlops.monitoring.retrain_trigger import evaluate_retrain_trigger


def test_scenario_1_healthy():
    """Scenario 1: Healthy system."""
    quality_metrics = {
        "n_records": 500,
        "potential_failure_rate": 0.05,
        "uncertain_rate": 0.1,
    }
    
    result = evaluate_retrain_trigger(
        data_drift_detected=False,
        routing_drift_detected=False,
        quality_drop_detected=False,
        quality_metrics=quality_metrics,
        config={"retrain": {"minimum_samples": 100}, "drift": {"quality": {"routing_failure_rate_threshold": 0.10}}}
    )
    
    assert result["overall_conclusion"] == "healthy"
    assert result["retrain_recommended"] is False


def test_scenario_2_user_behavior_change():
    """Scenario 2: Data drift, but no quality drop."""
    quality_metrics = {
        "n_records": 500,
        "potential_failure_rate": 0.08,
        "uncertain_rate": 0.1,
    }
    
    result = evaluate_retrain_trigger(
        data_drift_detected=True,
        routing_drift_detected=False,
        quality_drop_detected=False,
        quality_metrics=quality_metrics,
        config={"retrain": {"minimum_samples": 100}, "drift": {"quality": {"routing_failure_rate_threshold": 0.10}}}
    )
    
    assert result["overall_conclusion"] == "input_distribution_changed"
    assert result["retrain_recommended"] is False
    assert len(result["warnings"]) > 0
    assert "user behavior" in result["warnings"][0].lower()


def test_scenario_3_routing_problem():
    """Scenario 3: No data drift, but quality drops and failures spike."""
    quality_metrics = {
        "n_records": 500,
        "potential_failure_rate": 0.15,  # Above 0.10 threshold
        "uncertain_rate": 0.1,
    }
    
    result = evaluate_retrain_trigger(
        data_drift_detected=False,
        routing_drift_detected=False,
        quality_drop_detected=True,
        quality_metrics=quality_metrics,
        config={"retrain": {"minimum_samples": 100}, "drift": {"quality": {"routing_failure_rate_threshold": 0.10}}}
    )
    
    assert result["overall_conclusion"] == "response_quality_degradation"
    assert result["retrain_recommended"] is True
    assert "failure_rate=15.00%" in result["retrain_reason"]


def test_scenario_4_model_degradation():
    """Scenario 4: Everything drifts (Model Degradation)."""
    quality_metrics = {
        "n_records": 500,
        "potential_failure_rate": 0.20,  # Above 0.10 threshold
        "uncertain_rate": 0.1,
    }
    
    result = evaluate_retrain_trigger(
        data_drift_detected=True,
        routing_drift_detected=True,
        quality_drop_detected=True,
        quality_metrics=quality_metrics,
        config={"retrain": {"minimum_samples": 100}, "drift": {"quality": {"routing_failure_rate_threshold": 0.10}}}
    )
    
    assert result["overall_conclusion"] == "potential_model_degradation"
    assert result["retrain_recommended"] is True


def test_insufficient_samples():
    """Ensure retraining is NOT triggered if sample size is too small."""
    quality_metrics = {
        "n_records": 50,  # Below minimum 100
        "potential_failure_rate": 0.20,
        "uncertain_rate": 0.1,
    }
    
    result = evaluate_retrain_trigger(
        data_drift_detected=True,
        routing_drift_detected=True,
        quality_drop_detected=True,
        quality_metrics=quality_metrics,
        config={"retrain": {"minimum_samples": 100}, "drift": {"quality": {"routing_failure_rate_threshold": 0.10}}}
    )
    
    assert result["retrain_recommended"] is False
    assert len(result["warnings"]) > 0
    assert "insufficient samples" in result["warnings"][0].lower()
