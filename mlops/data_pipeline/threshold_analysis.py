"""
Threshold Sensitivity Analysis — Analyze how different quality thresholds
affect label distribution for the LLM Router.

UltraFeedback scores are on scale 1-5. This module tests multiple threshold
values (τ) and reports how label distribution changes, helping researchers
select an appropriate τ for their routing needs.

Expected behavior:
  τ = 3.0 → ~70% weak, ~20% medium, ~10% strong (very lenient)
  τ = 3.5 → ~50% weak, ~30% medium, ~20% strong
  τ = 4.0 → ~35% weak, ~35% medium, ~30% strong (recommended default)
  τ = 4.5 → ~15% weak, ~25% medium, ~60% strong (very strict)
"""

import json
from pathlib import Path
from typing import Optional
from collections import Counter

import yaml

from mlops.data_pipeline.score_aggregation import (
    compute_tier_scores,
    DEFAULT_WEIGHTS,
)
from mlops.data_pipeline.label import assign_minimum_sufficient_label


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_threshold_analysis(
    input_path: str = "data/processed/ultrafeedback_deduped.jsonl",
    config: Optional[dict] = None,
    thresholds: Optional[list] = None,
    margins: Optional[list] = None,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run sensitivity analysis across multiple threshold values.
    
    For each (threshold, margin) combination, labels all prompts and
    reports the resulting distribution.
    
    Args:
        input_path: Path to processed UltraFeedback data.
        config: Dataset config (for model tiers, weights).
        thresholds: List of τ values to test (scale 1-5).
        margins: List of margin values to test (scale 1-5).
        max_samples: Limit number of prompts for faster analysis.
        output_path: Path to save analysis report.
    
    Returns:
        Analysis report dict.
    """
    if config is None:
        config = load_config()
    
    labeling_cfg = config.get("labeling", {})
    model_tiers = config.get("model_tiers", {})
    
    if thresholds is None:
        thresholds = labeling_cfg.get("threshold_candidates", [3.0, 3.5, 4.0, 4.5])
    if margins is None:
        margins = [0.5, 0.75, 1.0]
    
    # Score aggregation config
    agg_cfg = labeling_cfg.get("score_aggregation", {})
    weights = agg_cfg.get("weights", DEFAULT_WEIGHTS)
    tier_score_method = labeling_cfg.get("tier_score_method", "best")
    fallback = agg_cfg.get("fallback", "available_mean")
    
    in_path = Path(input_path)
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print("=" * 70)
    print("  THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  Input: {input_path}")
    print(f"  Thresholds (tau): {thresholds}")
    print(f"  Margins: {margins}")
    print(f"  Max samples: {max_samples or 'all'}")
    print()
    
    # Pre-compute tier scores for all records
    print("  Computing tier scores for all records...")
    records_tier_scores = []
    
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            completions = record.get("completions", [])
            
            tier_scores = compute_tier_scores(
                completions, model_tiers, weights, tier_score_method, fallback
            )
            
            # Only include if at least one tier has scores
            has_scores = any(
                tier_scores[t]["score"] is not None
                for t in ["weak", "medium", "strong"]
            )
            if has_scores:
                records_tier_scores.append({
                    "weak": tier_scores["weak"]["score"],
                    "medium": tier_scores["medium"]["score"],
                    "strong": tier_scores["strong"]["score"],
                })
    
    n_records = len(records_tier_scores)
    print(f"  Records with scores: {n_records}")
    print()
    
    # Run analysis for each (threshold, margin) combination
    results = {}
    
    print(f"  {'tau':>5} {'margin':>7} | {'weak':>8} {'medium':>8} {'strong':>8} | {'weak%':>6} {'med%':>6} {'str%':>6}")
    print(f"  {'-'*5} {'-'*7} | {'-'*8} {'-'*8} {'-'*8} | {'-'*6} {'-'*6} {'-'*6}")
    
    for tau in thresholds:
        for margin in margins:
            label_counts = Counter()
            
            for ts in records_tier_scores:
                label = assign_minimum_sufficient_label(
                    weak_score=ts["weak"],
                    medium_score=ts["medium"],
                    strong_score=ts["strong"],
                    threshold=tau,
                    margin=margin,
                )
                label_counts[label] += 1
            
            key = f"tau={tau}_m={margin}"
            pcts = {
                l: round(label_counts.get(l, 0) / n_records * 100, 1) if n_records > 0 else 0
                for l in [0, 1, 2]
            }
            
            results[key] = {
                "threshold": tau,
                "margin": margin,
                "n_records": n_records,
                "label_distribution": dict(label_counts),
                "label_percentages": pcts,
            }
            
            print(f"  {tau:>5.1f} {margin:>7.2f} | "
                  f"{label_counts.get(0, 0):>8} {label_counts.get(1, 0):>8} {label_counts.get(2, 0):>8} | "
                  f"{pcts[0]:>5.1f}% {pcts[1]:>5.1f}% {pcts[2]:>5.1f}%")
    
    print()
    
    # Score distribution summary
    print("  Score Distribution Summary (across all records):")
    for tier_name in ["weak", "medium", "strong"]:
        scores = [ts[tier_name] for ts in records_tier_scores if ts[tier_name] is not None]
        if scores:
            import statistics
            print(f"    {tier_name:>8}: n={len(scores):>5}, "
                  f"mean={statistics.mean(scores):.2f}, "
                  f"std={statistics.stdev(scores):.2f}, "
                  f"min={min(scores):.2f}, max={max(scores):.2f}")
        else:
            print(f"    {tier_name:>8}: no data")
    
    print("=" * 70)
    
    # Build report
    report = {
        "analysis_type": "threshold_sensitivity",
        "score_scale": "1-5",
        "input_path": input_path,
        "n_records": n_records,
        "thresholds_tested": thresholds,
        "margins_tested": margins,
        "results": results,
    }
    
    # Save report
    if output_path is None:
        output_path = "data/metadata/threshold_analysis.json"
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Threshold Sensitivity Analysis")
    parser.add_argument("--input", type=str, default="data/processed/ultrafeedback_deduped.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/metadata/threshold_analysis.json")
    
    args = parser.parse_args()
    run_threshold_analysis(
        input_path=args.input,
        max_samples=args.max_samples,
        output_path=args.output,
    )
