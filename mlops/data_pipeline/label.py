"""
Routing Label Construction — Derive routing labels from model performance data.

Core principle:
  Routing label = which model tier is the MINIMUM SUFFICIENT for a given query,
  NOT which model scores highest.

Methods:
  1. minimum_sufficient_model (recommended):
     - Compute weighted quality score from 4 dimensions (scale 1-5)
     - Group models into tiers (weak/medium/strong)
     - Label = weakest tier that achieves quality threshold
     - Apply confidence margin to handle borderline cases
     
  2. ultrafeedback_score (legacy):
     - Use overall_score directly from UltraFeedback
     - Same minimum sufficient logic but without weighted aggregation
     
  3. mock_length (baseline only):
     - Simple query length heuristic
     - Used ONLY as experimental baseline, never as ground truth

IMPORTANT:
  - UltraFeedback scores are on scale 1-5 (NOT 1-10)
  - Each prompt has 4 completions sampled from a pool of 17 models
  - Not every prompt has all tiers represented
  - Labels are derived pseudo-labels, NOT ground truth
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

import yaml

from mlops.data_pipeline.score_aggregation import (
    aggregate_dimension_scores,
    compute_tier_scores,
    DEFAULT_WEIGHTS,
)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_model_tier(model_name: str, config: dict) -> Optional[str]:
    """Map a model name to its tier (weak/medium/strong)."""
    tiers = config.get("model_tiers", {})
    model_lower = model_name.lower().strip()
    
    for tier_name, models in tiers.items():
        for m in models:
            if m.lower() in model_lower or model_lower in m.lower():
                return tier_name
    
    return None  # Unknown model


def assign_minimum_sufficient_label(
    weak_score: Optional[float],
    medium_score: Optional[float],
    strong_score: Optional[float],
    threshold: float = 4.0,
    margin: float = 0.75,
) -> int:
    """
    Assign routing label based on Minimum Sufficient Model principle.
    
    Logic (scale 1-5):
      1. If weak_score >= threshold AND (strong_score - weak_score) < margin:
         → label = 0 (weak sufficient)
      2. Elif medium_score >= threshold AND (strong_score - medium_score) < margin:
         → label = 1 (medium needed)
      3. Else:
         → label = 2 (strong needed)
    
    The confidence margin prevents labeling a prompt as "weak sufficient"
    when a strong model would produce meaningfully better output.
    
    Args:
        weak_score: Best quality score from weak-tier models (1-5), or None.
        medium_score: Best quality score from medium-tier models (1-5), or None.
        strong_score: Best quality score from strong-tier models (1-5), or None.
        threshold: Minimum quality score to consider a tier "sufficient" (scale 1-5).
        margin: Score gap threshold. If strong - weak > margin, upscale label.
    
    Returns:
        Routing label: 0 (weak), 1 (medium), or 2 (strong).
    
    Examples (threshold=4.0, margin=0.75):
        >>> # Easy prompt: weak is good enough, small gap to strong
        >>> assign_minimum_sufficient_label(4.2, 4.5, 4.7, 4.0, 0.75)
        0  # weak sufficient (gap = 0.5 < 0.75)
        
        >>> # Weak passes threshold but strong is much better
        >>> assign_minimum_sufficient_label(4.1, 4.5, 5.0, 4.0, 0.75)
        1  # upscaled: gap = 0.9 > 0.75, but medium also OK
        
        >>> # Medium difficulty
        >>> assign_minimum_sufficient_label(3.5, 4.3, 4.6, 4.0, 0.75)
        1  # medium needed (weak below threshold)
        
        >>> # Hard prompt
        >>> assign_minimum_sufficient_label(2.0, 3.2, 4.5, 4.0, 0.75)
        2  # only strong meets threshold
    """
    # Handle None scores (tier not present for this prompt)
    ws = weak_score if weak_score is not None else 0.0
    ms = medium_score if medium_score is not None else 0.0
    ss = strong_score if strong_score is not None else 0.0
    
    # Compute score gaps
    gap_weak_strong = ss - ws
    gap_medium_strong = ss - ms
    
    # Minimum Sufficient Model with confidence margin
    if ws >= threshold and gap_weak_strong < margin:
        return 0  # Weak is sufficient
    elif ms >= threshold and gap_medium_strong < margin:
        return 1  # Medium is sufficient
    elif ws >= threshold:
        # Weak passes threshold but gap is large → upscale to medium
        return 1
    elif ms >= threshold:
        # Medium passes threshold but gap is large → upscale to strong
        return 2
    else:
        return 2  # Strong needed (or no tier meets threshold)


def label_ultrafeedback_msm(
    input_path: str = "data/processed/ultrafeedback_deduped.jsonl",
    output_path: str = "data/labeled/ultrafeedback_labeled.jsonl",
    config: Optional[dict] = None,
) -> dict:
    """
    Construct routing labels using Minimum Sufficient Model method.
    
    Algorithm:
      1. For each prompt, collect all completions with dimension scores
      2. Compute weighted quality score per completion (4 dims, scale 1-5)
      3. Group by model tier, take best score per tier
      4. Apply Minimum Sufficient Model logic with confidence margin
      
    Label interpretation:
      0 = weak model sufficient (easy prompt)
      1 = medium model needed (moderate prompt)
      2 = strong model needed (hard prompt)
    
    Returns:
        Labeling statistics dict.
    """
    if config is None:
        config = load_config()
    
    labeling_cfg = config.get("labeling", {})
    quality_threshold = labeling_cfg.get("quality_threshold", 4.0)
    confidence_margin = labeling_cfg.get("confidence_margin", 0.75)
    tier_score_method = labeling_cfg.get("tier_score_method", "best")
    missing_tier_strategy = labeling_cfg.get("missing_tier_strategy", "label_from_available")
    label_map = labeling_cfg.get("label_map", {"weak": 0, "medium": 1, "strong": 2})
    
    # Score aggregation config
    agg_cfg = labeling_cfg.get("score_aggregation", {})
    agg_method = agg_cfg.get("method", "weighted")
    weights = agg_cfg.get("weights", DEFAULT_WEIGHTS)
    fallback = agg_cfg.get("fallback", "available_mean")
    
    model_tiers = config.get("model_tiers", {})
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Labeling: {input_path}")
    print(f"  Method: minimum_sufficient_model")
    print(f"  Score scale: 1-5")
    print(f"  Quality threshold (tau): {quality_threshold}")
    print(f"  Confidence margin: {confidence_margin}")
    print(f"  Score aggregation: {agg_method}")
    print(f"  Tier score method: {tier_score_method}")
    print(f"  Missing tier strategy: {missing_tier_strategy}")
    
    total = 0
    labeled = 0
    skipped_no_score = 0
    skipped_missing_tier = 0
    label_counts = Counter()
    tier_coverage = Counter()
    tier_coverage_combinations = Counter()
    
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            record = json.loads(line)
            
            instruction = record.get("instruction", "")
            completions = record.get("completions", [])
            
            # Compute tier scores using weighted aggregation
            if agg_method == "overall_score_only":
                # Legacy: use overall_score directly
                tier_scores = _compute_tier_scores_legacy(completions, config)
            else:
                tier_scores = compute_tier_scores(
                    completions, model_tiers, weights, tier_score_method, fallback
                )
            
            # Check tier coverage
            tiers_present = []
            for tier_name in ["weak", "medium", "strong"]:
                if tier_scores[tier_name]["score"] is not None:
                    tiers_present.append(tier_name)
                    tier_coverage[tier_name] += 1
            
            tier_coverage_combinations[tuple(sorted(tiers_present))] += 1
            
            if not tiers_present:
                skipped_no_score += 1
                continue
            
            # Handle missing tiers
            if len(tiers_present) < 3 and missing_tier_strategy == "skip":
                skipped_missing_tier += 1
                continue
            
            # Extract best scores per tier
            weak_score = tier_scores["weak"]["score"]
            medium_score = tier_scores["medium"]["score"]
            strong_score = tier_scores["strong"]["score"]
            
            # Assign label using Minimum Sufficient Model
            routing_label = assign_minimum_sufficient_label(
                weak_score=weak_score,
                medium_score=medium_score,
                strong_score=strong_score,
                threshold=quality_threshold,
                margin=confidence_margin,
            )
            
            # Map label to config label values
            label_names = {0: "weak", 1: "medium", 2: "strong"}
            
            # Build label evidence
            score_gap_ws = ((strong_score or 0) - (weak_score or 0)) if weak_score is not None else None
            score_gap_ms = ((strong_score or 0) - (medium_score or 0)) if medium_score is not None else None
            
            labeled_record = {
                "prompt": instruction,
                "routing_label": routing_label,
                "label_type": "derived_routing_label",
                "label_method": "minimum_sufficient_model",
                "label_evidence": {
                    "weak_quality_score": weak_score,
                    "medium_quality_score": medium_score,
                    "strong_quality_score": strong_score,
                    "quality_threshold": quality_threshold,
                    "confidence_margin": confidence_margin,
                    "score_gap_weak_strong": round(score_gap_ws, 4) if score_gap_ws is not None else None,
                    "score_gap_medium_strong": round(score_gap_ms, 4) if score_gap_ms is not None else None,
                    "tier_coverage": {t: tier_scores[t]["n_models"] for t in ["weak", "medium", "strong"]},
                    "weak_models": tier_scores["weak"]["models"],
                    "medium_models": tier_scores["medium"]["models"],
                    "strong_models": tier_scores["strong"]["models"],
                    "minimum_sufficient_tier": label_names[routing_label],
                },
                "prompt_length": len(instruction),
                "prompt_word_count": len(instruction.split()),
                "source": record.get("source", ""),
            }
            
            fout.write(json.dumps(labeled_record, ensure_ascii=False) + "\n")
            labeled += 1
            label_counts[routing_label] += 1
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "method": "minimum_sufficient_model",
        "score_scale": "1-5",
        "quality_threshold": quality_threshold,
        "confidence_margin": confidence_margin,
        "score_aggregation": agg_method,
        "total_input": total,
        "total_labeled": labeled,
        "skipped_no_score": skipped_no_score,
        "skipped_missing_tier": skipped_missing_tier,
        "label_distribution": dict(label_counts),
        "tier_coverage": dict(tier_coverage),
        "tier_coverage_combinations": {str(k): v for k, v in tier_coverage_combinations.most_common()},
    }
    
    print(f"\nLabeling complete!")
    print(f"  Input:          {total}")
    print(f"  Labeled:        {labeled}")
    print(f"  Skipped (score):{skipped_no_score}")
    print(f"  Skipped (tier): {skipped_missing_tier}")
    print(f"\n  Label Distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / labeled * 100 if labeled > 0 else 0
        names = {0: "weak", 1: "medium", 2: "strong"}
        print(f"    {label} ({names.get(label, '?')}): {count} ({pct:.1f}%)")
    print(f"\n  Tier Coverage (prompts with at least 1 model in tier):")
    for tier, count in tier_coverage.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"    {tier}: {count} ({pct:.1f}%)")
    print(f"\n  Top Tier Combinations:")
    for combo, count in list(tier_coverage_combinations.most_common(5)):
        print(f"    {combo}: {count}")
    
    # Save stats
    meta_path = Path(output_path).parent.parent / "metadata" / "labeling_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def _compute_tier_scores_legacy(completions: list, config: dict) -> dict:
    """Legacy tier score computation using overall_score directly."""
    model_tiers = config.get("model_tiers", {})
    
    # Build reverse mapping
    model_to_tier = {}
    for tier_name, models in model_tiers.items():
        for m in models:
            model_to_tier[m.lower().strip()] = tier_name
    
    tier_data = {
        "weak": {"scores": [], "models": []},
        "medium": {"scores": [], "models": []},
        "strong": {"scores": [], "models": []},
    }
    
    for comp in completions:
        model = comp.get("model", "").lower().strip()
        score = comp.get("overall_score")
        
        if score is None:
            continue
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        
        tier = model_to_tier.get(model)
        if tier is None:
            for known, t in model_to_tier.items():
                if known in model or model in known:
                    tier = t
                    break
        
        if tier and tier in tier_data:
            tier_data[tier]["scores"].append(score)
            tier_data[tier]["models"].append(comp.get("model", ""))
    
    result = {}
    for tier_name in ["weak", "medium", "strong"]:
        data = tier_data[tier_name]
        if data["scores"]:
            result[tier_name] = {
                "score": round(max(data["scores"]), 4),
                "n_models": len(data["scores"]),
                "models": data["models"],
                "all_scores": [round(s, 4) for s in data["scores"]],
                "best_dimension_scores": {},
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


def label_ultrafeedback(
    input_path: str = "data/processed/ultrafeedback_deduped.jsonl",
    output_path: str = "data/labeled/ultrafeedback_labeled.jsonl",
    config: Optional[dict] = None,
) -> dict:
    """
    Construct routing labels from UltraFeedback scores.
    
    This is the main entry point. It delegates to the appropriate method
    based on the config's labeling.method setting.
    
    For backward compatibility, this function supports both:
      - "minimum_sufficient_model" (recommended, uses weighted dimension scores)
      - "ultrafeedback_score" (legacy, uses overall_score with MSM logic)
    
    Returns:
        Labeling statistics dict.
    """
    if config is None:
        config = load_config()
    
    method = config.get("labeling", {}).get("method", "minimum_sufficient_model")
    
    if method == "minimum_sufficient_model":
        return label_ultrafeedback_msm(input_path, output_path, config)
    elif method == "ultrafeedback_score":
        # Legacy method but now with corrected threshold (scale 1-5)
        return label_ultrafeedback_msm(input_path, output_path, config)
    else:
        return {"error": f"Unknown labeling method: {method}"}


def label_mock_baseline(
    input_path: str = "data/processed/ultrafeedback_deduped.jsonl",
    output_path: str = "data/labeled/mock_labeled.jsonl",
    config: Optional[dict] = None,
) -> dict:
    """
    Create MOCK labels based on query length.
    
    THIS IS A BASELINE ONLY — NOT a valid labeling method.
    Used in experiments to demonstrate the limitation of
    heuristic-based labeling vs. model-capability-based labeling.
    
    Returns:
        Labeling statistics dict.
    """
    if config is None:
        config = load_config()
    
    mock_cfg = config.get("labeling", {}).get("mock", {})
    short_threshold = mock_cfg.get("short_threshold", 50)
    medium_threshold = mock_cfg.get("medium_threshold", 150)
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Creating MOCK baseline labels: {input_path}")
    print(f"  WARNING: These are HEURISTIC labels, NOT ground truth!")
    
    total = 0
    label_counts = Counter()
    
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            record = json.loads(line)
            instruction = record.get("instruction", "")
            
            # Mock labeling based on length
            prompt_len = len(instruction)
            if prompt_len < short_threshold:
                routing_label = 0
            elif prompt_len < medium_threshold:
                routing_label = 1
            else:
                routing_label = 2
            
            labeled_record = {
                "prompt": instruction,
                "routing_label": routing_label,
                "label_type": "mock_label",  # CLEARLY MARKED as mock
                "label_method": "query_length_heuristic",
                "label_evidence": {
                    "prompt_length": prompt_len,
                    "short_threshold": short_threshold,
                    "medium_threshold": medium_threshold,
                },
                "prompt_length": prompt_len,
                "prompt_word_count": len(instruction.split()),
                "source": record.get("source", ""),
            }
            
            fout.write(json.dumps(labeled_record, ensure_ascii=False) + "\n")
            label_counts[routing_label] += 1
    
    stats = {
        "method": "mock_length",
        "total_labeled": total,
        "label_distribution": dict(label_counts),
        "note": "MOCK LABELS — heuristic baseline only, NOT ground truth",
    }
    
    print(f"  Total: {total}")
    print(f"  Distribution: {dict(label_counts)}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct routing labels")
    parser.add_argument("--method", type=str, default="minimum_sufficient_model",
                        choices=["minimum_sufficient_model", "ultrafeedback_score", "mock_length", "all"])
    parser.add_argument("--input", type=str, default="data/processed/ultrafeedback_deduped.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/labeled")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    
    if args.method in ("minimum_sufficient_model", "ultrafeedback_score", "all"):
        label_ultrafeedback(args.input, f"{args.output_dir}/ultrafeedback_labeled.jsonl", config)
    
    if args.method in ("mock_length", "all"):
        label_mock_baseline(args.input, f"{args.output_dir}/mock_labeled.jsonl", config)
