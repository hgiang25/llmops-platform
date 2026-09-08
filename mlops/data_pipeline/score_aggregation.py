"""
Score Aggregation — Compute quality scores from UltraFeedback multi-dimension annotations.

UltraFeedback provides 4 dimension scores per completion, each on scale 1-5:
  - helpfulness: How useful and relevant the response is
  - truthfulness: How factually accurate the response is
  - honesty: How well the model expresses uncertainty when appropriate
  - instruction_following: How well the response follows the user's instructions

This module provides weighted aggregation of these dimensions into a single
quality score, handling missing dimensions gracefully.

Score Scale Reference (1-5):
  1 = Very Poor
  2 = Poor
  3 = Acceptable
  4 = Good
  5 = Excellent
"""

from typing import Optional


# Default weights for score aggregation
DEFAULT_WEIGHTS = {
    "helpfulness": 0.30,
    "instruction_following": 0.25,
    "truthfulness": 0.25,
    "honesty": 0.20,
}

SCORE_DIMENSIONS = ["helpfulness", "instruction_following", "truthfulness", "honesty"]
SCORE_MIN = 1.0
SCORE_MAX = 5.0


def aggregate_dimension_scores(
    scores: dict,
    weights: Optional[dict] = None,
    fallback: str = "available_mean",
) -> Optional[float]:
    """
    Compute weighted quality score from dimension scores.
    
    Args:
        scores: Dict of dimension_name -> score (scale 1-5).
                Example: {"helpfulness": 4.0, "truthfulness": 3.0, "honesty": 5.0}
        weights: Dict of dimension_name -> weight (should sum to 1.0).
                 If None, uses DEFAULT_WEIGHTS.
        fallback: How to handle missing dimensions.
                  "available_mean": redistribute weight proportionally among available dims.
                  "skip": return None if any dimension is missing.
                  "default_3": use 3.0 (midpoint) for missing dimensions.
    
    Returns:
        Aggregated quality score on scale 1-5, or None if not computable.
    
    Example:
        >>> aggregate_dimension_scores(
        ...     {"helpfulness": 4.0, "truthfulness": 3.0, "honesty": 5.0, "instruction_following": 4.0}
        ... )
        3.95  # = 0.30*4.0 + 0.25*4.0 + 0.25*3.0 + 0.20*5.0
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    if not scores:
        return None
    
    # Collect available dimension scores
    available = {}
    for dim in SCORE_DIMENSIONS:
        if dim in scores and scores[dim] is not None:
            try:
                val = float(scores[dim])
                if SCORE_MIN <= val <= SCORE_MAX:
                    available[dim] = val
            except (ValueError, TypeError):
                continue
    
    if not available:
        return None
    
    if fallback == "skip" and len(available) < len(SCORE_DIMENSIONS):
        return None
    
    if fallback == "default_3":
        # Fill missing dimensions with midpoint (3.0)
        for dim in SCORE_DIMENSIONS:
            if dim not in available:
                available[dim] = 3.0
    
    # Compute weighted score with available dimensions
    # Redistribute weights proportionally if dimensions are missing
    available_weight_sum = sum(weights.get(dim, 0) for dim in available)
    if available_weight_sum <= 0:
        return None
    
    weighted_score = 0.0
    for dim, val in available.items():
        w = weights.get(dim, 0)
        # Redistribute: normalize weight relative to available dimensions
        normalized_w = w / available_weight_sum
        weighted_score += normalized_w * val
    
    return round(weighted_score, 4)


def compute_tier_scores(
    completions: list[dict],
    model_tiers: dict,
    weights: Optional[dict] = None,
    method: str = "best",
    fallback: str = "available_mean",
) -> dict:
    """
    Compute quality scores per tier from a prompt's completions.
    
    Args:
        completions: List of completion dicts, each with "model", "scores", "overall_score".
        model_tiers: Dict mapping tier_name -> list of model names.
        weights: Score aggregation weights.
        method: "best" (max score per tier) or "mean" (average per tier).
        fallback: How to handle missing dimensions.
    
    Returns:
        Dict with tier scores and metadata:
        {
            "weak": {"score": 3.5, "n_models": 2, "models": ["alpaca-7b", "wizardlm-7b"]},
            "medium": {"score": 4.2, "n_models": 1, "models": ["llama-2-13b-chat"]},
            "strong": {"score": None, "n_models": 0, "models": []},
        }
    """
    # Build reverse mapping: model_name -> tier
    model_to_tier = {}
    for tier_name, models in model_tiers.items():
        for m in models:
            model_to_tier[m.lower().strip()] = tier_name
    
    # Collect scores per tier
    tier_data = {
        "weak": {"scores": [], "models": [], "dimension_scores": []},
        "medium": {"scores": [], "models": [], "dimension_scores": []},
        "strong": {"scores": [], "models": [], "dimension_scores": []},
    }
    
    for comp in completions:
        model = comp.get("model", "").lower().strip()
        
        # Find tier for this model (fuzzy matching)
        tier = None
        if model in model_to_tier:
            tier = model_to_tier[model]
        else:
            # Try partial matching
            for known_model, t in model_to_tier.items():
                if known_model in model or model in known_model:
                    tier = t
                    break
        
        if tier is None or tier not in tier_data:
            continue
        
        # Compute quality score from dimensions
        dim_scores = comp.get("scores", {})
        quality_score = aggregate_dimension_scores(dim_scores, weights, fallback)
        
        # Fallback to overall_score if dimension aggregation fails
        if quality_score is None:
            overall = comp.get("overall_score")
            if overall is not None:
                try:
                    quality_score = float(overall)
                except (ValueError, TypeError):
                    continue
        
        if quality_score is not None:
            tier_data[tier]["scores"].append(quality_score)
            tier_data[tier]["models"].append(comp.get("model", ""))
            tier_data[tier]["dimension_scores"].append(dim_scores)
    
    # Compute representative score per tier
    result = {}
    for tier_name in ["weak", "medium", "strong"]:
        data = tier_data[tier_name]
        if data["scores"]:
            if method == "best":
                score = max(data["scores"])
            else:  # mean
                score = sum(data["scores"]) / len(data["scores"])
            
            # Find the dimension scores of the best model
            best_idx = data["scores"].index(max(data["scores"]))
            best_dim_scores = data["dimension_scores"][best_idx] if data["dimension_scores"] else {}
            
            result[tier_name] = {
                "score": round(score, 4),
                "n_models": len(data["scores"]),
                "models": data["models"],
                "all_scores": [round(s, 4) for s in data["scores"]],
                "best_dimension_scores": best_dim_scores,
            }
        else:
            result[tier_name] = {
                "score": None,
                "n_models": 0,
                "models": [],
                "all_scores": [],
                "best_dimension_scores": {},
            }
    
    return result


def analyze_score_distribution(records: list[dict], model_tiers: dict, weights: Optional[dict] = None) -> dict:
    """
    Analyze score distribution across tiers for a collection of records.
    
    Args:
        records: List of raw UltraFeedback records with "completions".
        model_tiers: Model tier mapping.
        weights: Score aggregation weights.
    
    Returns:
        Distribution statistics per tier: mean, std, percentiles, histograms.
    """
    import statistics
    
    tier_all_scores = {"weak": [], "medium": [], "strong": []}
    
    for record in records:
        completions = record.get("completions", [])
        tier_scores = compute_tier_scores(completions, model_tiers, weights)
        
        for tier_name in ["weak", "medium", "strong"]:
            all_s = tier_scores[tier_name].get("all_scores", [])
            tier_all_scores[tier_name].extend(all_s)
    
    result = {}
    for tier_name, scores in tier_all_scores.items():
        if scores:
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            result[tier_name] = {
                "count": n,
                "mean": round(statistics.mean(scores), 4),
                "std": round(statistics.stdev(scores), 4) if n > 1 else 0,
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "p25": round(sorted_scores[int(n * 0.25)], 4),
                "p50": round(sorted_scores[int(n * 0.50)], 4),
                "p75": round(sorted_scores[int(n * 0.75)], 4),
                "histogram": _score_histogram(scores),
            }
        else:
            result[tier_name] = {"count": 0}
    
    return result


def _score_histogram(scores: list[float], bins: int = 5) -> dict:
    """Create histogram with 5 bins for 1-5 scale: [1-2), [2-3), [3-4), [4-5), [5]."""
    histogram = {"1.0-2.0": 0, "2.0-3.0": 0, "3.0-4.0": 0, "4.0-5.0": 0, "5.0": 0}
    for s in scores:
        if s < 2.0:
            histogram["1.0-2.0"] += 1
        elif s < 3.0:
            histogram["2.0-3.0"] += 1
        elif s < 4.0:
            histogram["3.0-4.0"] += 1
        elif s < 5.0:
            histogram["4.0-5.0"] += 1
        else:
            histogram["5.0"] += 1
    return histogram
