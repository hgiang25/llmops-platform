"""
Unit tests for the MLOps Batch Scorer pipeline.
Tests Semantic KNN, Feedback scoring, and Score Aggregation.
"""

import pytest

from mlops.scoring.semantic_knn import SemanticKNN
from mlops.scoring.feedback_scorer import compute_feedback_signal
from mlops.scoring.score_aggregator import aggregate_scores


def test_feedback_scorer():
    """Test feedback score computation based on signals."""
    # Positive feedback
    assert compute_feedback_signal(thumbs_up=True)["feedback_signal"] == "positive"
    
    # Negative feedback
    assert compute_feedback_signal(thumbs_down=True)["feedback_signal"] == "negative"
    assert compute_feedback_signal(regenerate=True)["feedback_signal"] == "negative"
    
    # Strong negative (both)
    assert compute_feedback_signal(thumbs_down=True, regenerate=True)["feedback_signal"] == "strong_negative"
    
    # Conflicting or empty
    assert compute_feedback_signal(thumbs_up=True, thumbs_down=True)["feedback_signal"] == "absent"
    assert compute_feedback_signal()["feedback_signal"] == "absent"


def test_score_aggregator():
    """Test the aggregation of KNN and Feedback scores."""
    
    # 1. Base case: No feedback
    knn_result = {
        "assessment": "correct",
        "knn_confidence": 0.8,
        "knn_quality_estimate": 4.5,
        "is_uncertain": False,
    }
    agg = aggregate_scores(knn_result, {"feedback_signal": "absent", "has_feedback": False})
    assert agg["final_quality_score"] == 4.5
    
    # 2. Positive feedback on a correct route (boosts score)
    agg = aggregate_scores(knn_result, {"feedback_signal": "positive", "has_feedback": True})
    assert agg["final_quality_score"] >= 4.5
    
    # 3. Negative feedback on a correct route (downgrades score and changes assessment)
    agg = aggregate_scores(knn_result, {"feedback_signal": "negative", "has_feedback": True})
    assert agg["final_quality_score"] <= 2.5
    assert agg["routing_assessment"] == "potential_failure"
    
    # 4. Unknown assessment with no feedback
    knn_unknown = {
        "assessment": "unknown",
        "knn_confidence": 0.3,
        "knn_quality_estimate": 3.0,
        "is_uncertain": True,
    }
    agg = aggregate_scores(knn_unknown, {"feedback_signal": "absent", "has_feedback": False})
    assert agg["routing_assessment"] == "insufficient_evidence"
    
    # 5. Unknown assessment but positive feedback (gives benefit of the doubt)
    agg = aggregate_scores(knn_unknown, {"feedback_signal": "positive", "has_feedback": True})
    assert agg["routing_assessment"] == "correct"
    
    # 6. Unknown assessment with negative feedback (confirms failure)
    agg = aggregate_scores(knn_unknown, {"feedback_signal": "negative", "has_feedback": True})
    assert agg["routing_assessment"] == "potential_failure"
