"""
Routing-Specific Metrics — Beyond classification accuracy.

Evaluates routing decisions from a cost/quality/latency perspective:
  - Cost savings ratio
  - Unnecessary strong model calls  
  - Weak model failure rate
  - Routing utility (quality - λ*cost - μ*latency)
"""

import json
from pathlib import Path
from typing import Optional
from collections import Counter

import yaml


def load_eval_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "evaluation.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_routing_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    config: Optional[dict] = None,
) -> dict:
    """
    Compute routing-specific metrics beyond classification accuracy.
    
    Args:
        true_labels: Ground truth routing labels (0=weak, 1=medium, 2=strong)
        pred_labels: Predicted routing labels
        config: Evaluation config with cost model
    
    Returns:
        Dict with routing metrics.
    """
    if config is None:
        config = load_eval_config()
    
    cost_model = config.get("cost_model", {})
    
    costs = {
        0: cost_model.get("weak", {}).get("cost_per_token", 0.0001),
        1: cost_model.get("medium", {}).get("cost_per_token", 0.001),
        2: cost_model.get("strong", {}).get("cost_per_token", 0.01),
    }
    latencies = {
        0: cost_model.get("weak", {}).get("avg_latency_ms", 100),
        1: cost_model.get("medium", {}).get("avg_latency_ms", 300),
        2: cost_model.get("strong", {}).get("avg_latency_ms", 1000),
    }
    
    n = len(true_labels)
    if n == 0:
        return {"error": "No data"}
    
    # 1. Cost analysis
    # What would it cost to always use strong model?
    always_strong_cost = n * costs[2]
    # What does the router's routing cost?
    router_cost = sum(costs.get(p, costs[2]) for p in pred_labels)
    # What is the optimal cost (using true labels)?
    optimal_cost = sum(costs.get(t, costs[2]) for t in true_labels)
    
    cost_savings_vs_strong = 1 - (router_cost / always_strong_cost) if always_strong_cost > 0 else 0
    cost_efficiency = 1 - (router_cost / optimal_cost) if optimal_cost > 0 else 0  # Lower is better
    
    # 2. Unnecessary strong model calls
    # Router sent to strong but weak/medium would have been sufficient
    unnecessary_strong = sum(
        1 for t, p in zip(true_labels, pred_labels)
        if p == 2 and t < 2  # Predicted strong but didn't need it
    )
    unnecessary_strong_rate = unnecessary_strong / n
    
    # 3. Weak model failure rate
    # Router sent to weak but query actually needed medium/strong
    weak_failures = sum(
        1 for t, p in zip(true_labels, pred_labels)
        if p == 0 and t > 0  # Predicted weak but needed stronger
    )
    weak_failure_rate = weak_failures / n
    
    # 4. Under-routing (sent to weaker model than needed)
    under_routed = sum(1 for t, p in zip(true_labels, pred_labels) if p < t)
    under_routing_rate = under_routed / n
    
    # 5. Over-routing (sent to stronger model than needed — wasteful but not quality-damaging)
    over_routed = sum(1 for t, p in zip(true_labels, pred_labels) if p > t)
    over_routing_rate = over_routed / n
    
    # 6. Latency analysis
    always_strong_latency = n * latencies[2]
    router_latency = sum(latencies.get(p, latencies[2]) for p in pred_labels)
    optimal_latency = sum(latencies.get(t, latencies[2]) for t in true_labels)
    
    latency_savings_vs_strong = 1 - (router_latency / always_strong_latency) if always_strong_latency > 0 else 0
    
    # 7. Correct routing rate per class
    per_class_correct = {}
    for label in sorted(set(true_labels)):
        mask = [i for i, t in enumerate(true_labels) if t == label]
        correct = sum(1 for i in mask if pred_labels[i] == label)
        per_class_correct[label] = round(correct / len(mask), 4) if mask else 0
    
    # 8. Ordinal MAE (Mean Absolute Error)
    # For ordinal data, MAE captures that predicting 0 when truth is 2
    # is worse (error=2) than predicting 1 when truth is 2 (error=1)
    ordinal_mae = sum(abs(t - p) for t, p in zip(true_labels, pred_labels)) / n
    
    # 9. Severity-weighted error
    # Under-routing by 2 levels (0→2) is much worse than by 1 level (0→1)
    # Weight: (true - pred)^2 for under-routing, (pred - true) for over-routing
    severity_errors = []
    for t, p in zip(true_labels, pred_labels):
        if p < t:  # Under-routing (dangerous)
            severity_errors.append((t - p) ** 2)
        elif p > t:  # Over-routing (wasteful but safe)
            severity_errors.append(t - p)  # negative
        else:
            severity_errors.append(0)
    severity_weighted_error = sum(max(0, e) for e in severity_errors) / n
    
    # 10. Quality-Cost Utility
    # Utility = Quality - λ×Cost
    # Quality: 1.0 if correct or over-routed, penalty for under-routing
    lambda_cost = 0.5  # Cost weight
    quality_scores = []
    for t, p in zip(true_labels, pred_labels):
        if p >= t:
            quality_scores.append(1.0)  # Correct or over-routed → full quality
        else:
            quality_scores.append(max(0, 1.0 - (t - p) * 0.3))  # Penalty per level
    
    avg_quality = sum(quality_scores) / n
    normalized_cost = router_cost / always_strong_cost if always_strong_cost > 0 else 1.0
    utility = avg_quality - lambda_cost * normalized_cost
    
    metrics = {
        "n_samples": n,
        # Cost
        "cost_savings_vs_always_strong": round(cost_savings_vs_strong, 4),
        "router_total_cost": round(router_cost, 4),
        "optimal_total_cost": round(optimal_cost, 4),
        "always_strong_total_cost": round(always_strong_cost, 4),
        # Quality
        "unnecessary_strong_calls": unnecessary_strong,
        "unnecessary_strong_rate": round(unnecessary_strong_rate, 4),
        "weak_failures": weak_failures,
        "weak_failure_rate": round(weak_failure_rate, 4),
        "under_routing_rate": round(under_routing_rate, 4),
        "over_routing_rate": round(over_routing_rate, 4),
        # Ordinal
        "ordinal_mae": round(ordinal_mae, 4),
        "severity_weighted_error": round(severity_weighted_error, 4),
        # Utility
        "quality_cost_utility": round(utility, 4),
        "avg_quality": round(avg_quality, 4),
        "normalized_cost": round(normalized_cost, 4),
        # Latency
        "latency_savings_vs_always_strong": round(latency_savings_vs_strong, 4),
        "router_avg_latency_ms": round(router_latency / n, 2),
        "optimal_avg_latency_ms": round(optimal_latency / n, 2),
        # Per-class
        "per_class_correct_routing": per_class_correct,
        # Distribution
        "predicted_distribution": dict(Counter(pred_labels)),
        "true_distribution": dict(Counter(true_labels)),
    }
    
    return metrics


def print_routing_report(metrics: dict):
    """Print a formatted routing metrics report."""
    print("\n" + "=" * 60)
    print("ROUTING METRICS REPORT")
    print("=" * 60)
    
    print(f"\n  Samples: {metrics['n_samples']}")
    
    print(f"\n  --- Cost Analysis ---")
    print(f"  Cost savings vs always-strong: {metrics['cost_savings_vs_always_strong'] * 100:.1f}%")
    print(f"  Router cost:    {metrics['router_total_cost']:.4f}")
    print(f"  Optimal cost:   {metrics['optimal_total_cost']:.4f}")
    
    print(f"\n  --- Quality Analysis ---")
    print(f"  Weak failures:           {metrics['weak_failures']} ({metrics['weak_failure_rate'] * 100:.1f}%)")
    print(f"  Unnecessary strong calls: {metrics['unnecessary_strong_calls']} ({metrics['unnecessary_strong_rate'] * 100:.1f}%)")
    print(f"  Under-routing rate:      {metrics['under_routing_rate'] * 100:.1f}%")
    print(f"  Over-routing rate:       {metrics['over_routing_rate'] * 100:.1f}%")
    
    print(f"\n  --- Ordinal Metrics ---")
    print(f"  Ordinal MAE:             {metrics.get('ordinal_mae', 'N/A')}")
    print(f"  Severity-weighted error: {metrics.get('severity_weighted_error', 'N/A')}")
    
    print(f"\n  --- Quality-Cost Utility ---")
    print(f"  Utility (Q - λ×C):       {metrics.get('quality_cost_utility', 'N/A')}")
    print(f"  Avg Quality:             {metrics.get('avg_quality', 'N/A')}")
    print(f"  Normalized Cost:         {metrics.get('normalized_cost', 'N/A')}")
    
    print(f"\n  --- Latency Analysis ---")
    print(f"  Latency savings vs always-strong: {metrics['latency_savings_vs_always_strong'] * 100:.1f}%")
    print(f"  Router avg latency:   {metrics['router_avg_latency_ms']:.0f}ms")
    print(f"  Optimal avg latency:  {metrics['optimal_avg_latency_ms']:.0f}ms")
    
    print(f"\n  --- Per-Class Correct Routing ---")
    names = {0: "weak", 1: "medium", 2: "strong"}
    for label, rate in metrics.get("per_class_correct_routing", {}).items():
        print(f"    Class {label} ({names.get(int(label), '?')}): {rate * 100:.1f}%")
    
    print("=" * 60)

