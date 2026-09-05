"""
Dataset Quality Report — Generate comprehensive dataset analysis.

Usage:
    python scripts/dataset_report.py
    python scripts/dataset_report.py --input data/labeled/ultrafeedback_labeled.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def generate_report(input_path: str = "data/labeled/ultrafeedback_labeled.jsonl") -> dict:
    """Generate a comprehensive dataset quality report."""
    
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: File not found: {input_path}")
        return {"error": f"File not found: {input_path}"}
    
    print(f"Analyzing: {input_path}")
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    total = len(records)
    if total == 0:
        return {"error": "Empty dataset"}
    
    # Basic stats
    prompts = [r.get("prompt", "") for r in records]
    labels = [r.get("routing_label", -1) for r in records]
    label_types = [r.get("label_type", "unknown") for r in records]
    sources = [r.get("source", "unknown") for r in records]
    
    # Unique prompts
    unique_prompts = len(set(p.lower().strip() for p in prompts))
    duplicate_ratio = round(1 - unique_prompts / total, 4) if total > 0 else 0
    
    # Prompt lengths
    prompt_lengths = [len(p) for p in prompts]
    prompt_word_counts = [len(p.split()) for p in prompts]
    
    # Label distribution
    label_dist = dict(Counter(labels))
    label_type_dist = dict(Counter(label_types))
    
    # Source distribution
    source_dist = dict(Counter(sources))
    
    # Missing values
    missing_prompt = sum(1 for p in prompts if not p)
    missing_label = sum(1 for l in labels if l == -1)
    
    # Evidence analysis (for ultrafeedback labels)
    score_stats = {"weak": [], "medium": [], "strong": []}
    for r in records:
        evidence = r.get("label_evidence", {})
        if "best_weak_score" in evidence:
            score_stats["weak"].append(evidence["best_weak_score"])
        if "best_medium_score" in evidence:
            score_stats["medium"].append(evidence["best_medium_score"])
        if "best_strong_score" in evidence:
            score_stats["strong"].append(evidence["best_strong_score"])
    
    report = {
        "file": input_path,
        "total_samples": total,
        "unique_samples": unique_prompts,
        "duplicate_ratio": duplicate_ratio,
        # Prompt statistics
        "avg_prompt_length": round(sum(prompt_lengths) / total, 1),
        "min_prompt_length": min(prompt_lengths),
        "max_prompt_length": max(prompt_lengths),
        "median_prompt_length": sorted(prompt_lengths)[total // 2],
        "avg_prompt_words": round(sum(prompt_word_counts) / total, 1),
        # Label distribution
        "label_distribution": label_dist,
        "label_type_distribution": label_type_dist,
        # Source
        "source_distribution": dict(sorted(source_dist.items(), key=lambda x: -x[1])[:20]),
        # Missing
        "missing_prompt": missing_prompt,
        "missing_label": missing_label,
        # Score evidence
        "score_evidence": {
            tier: {
                "count": len(scores),
                "mean": round(sum(scores) / len(scores), 2) if scores else None,
                "min": round(min(scores), 2) if scores else None,
                "max": round(max(scores), 2) if scores else None,
            }
            for tier, scores in score_stats.items()
        },
    }
    
    # Print report
    print(f"\n{'=' * 60}")
    print("DATASET QUALITY REPORT")
    print(f"{'=' * 60}")
    print(f"  Total samples:       {total}")
    print(f"  Unique samples:      {unique_prompts}")
    print(f"  Duplicate ratio:     {duplicate_ratio * 100:.1f}%")
    print(f"\n  --- Prompt Statistics ---")
    print(f"  Avg length (chars):  {report['avg_prompt_length']}")
    print(f"  Min length:          {report['min_prompt_length']}")
    print(f"  Max length:          {report['max_prompt_length']}")
    print(f"  Avg words:           {report['avg_prompt_words']}")
    
    print(f"\n  --- Label Distribution ---")
    names = {0: "weak", 1: "medium", 2: "strong"}
    for label, count in sorted(label_dist.items()):
        pct = count / total * 100
        print(f"  Class {label} ({names.get(label, '?')}): {count} ({pct:.1f}%)")
    
    print(f"\n  --- Label Type ---")
    for lt, count in label_type_dist.items():
        print(f"  {lt}: {count}")
    
    print(f"\n  --- Score Evidence ---")
    for tier, stats in report["score_evidence"].items():
        if stats["count"] > 0:
            print(f"  {tier}: mean={stats['mean']}, range=[{stats['min']}, {stats['max']}]")
    
    print(f"\n  --- Missing Values ---")
    print(f"  Missing prompt: {missing_prompt}")
    print(f"  Missing label:  {missing_label}")
    
    print(f"\n  --- Top Sources ---")
    for src, count in list(report["source_distribution"].items())[:10]:
        print(f"  {src}: {count}")
    
    print(f"{'=' * 60}")
    
    # Save report
    report_dir = Path("reports/dataset_quality")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dataset_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate dataset quality report")
    parser.add_argument("--input", type=str, default="data/labeled/ultrafeedback_labeled.jsonl")
    
    args = parser.parse_args()
    generate_report(args.input)


if __name__ == "__main__":
    main()
