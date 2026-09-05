"""
Routing Label Construction — Derive routing labels from model performance data.

Core principle:
  Routing label = which model tier is SUFFICIENT for a given query,
  NOT which model scores highest.

Methods:
  1. ultrafeedback_score:
     - Use GPT-4 scores from UltraFeedback
     - Classify models into tiers (weak/medium/strong)
     - Label = weakest tier that achieves quality threshold
     
  2. chatbot_arena_pairwise:
     - Use human preference battle outcomes
     - Cluster models into tiers by Elo rating
     - Label based on whether weak model wins/ties against strong
     
  3. mock_length (baseline only):
     - Simple query length heuristic
     - Used ONLY as experimental baseline, never as ground truth

IMPORTANT DISTINCTIONS:
  - "mock_label": heuristic/synthetic label (baseline only)
  - "derived_routing_label": algorithmically derived from model scores
  - "human_preference_label": derived from human pairwise preference
  - These are NOT "ground_truth" — they are derived labels with known limitations
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

import yaml


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


def label_ultrafeedback(
    input_path: str = "data/processed/ultrafeedback_deduped.jsonl",
    output_path: str = "data/labeled/ultrafeedback_labeled.jsonl",
    config: Optional[dict] = None,
) -> dict:
    """
    Construct routing labels from UltraFeedback scores.
    
    Algorithm:
      1. For each prompt, collect all completions with scores
      2. Group completions by model tier (weak/medium/strong)
      3. Compute best score per tier
      4. Label = weakest tier that meets quality threshold
      
    Label interpretation:
      0 = weak model sufficient (best weak score >= threshold)
      1 = medium model needed (weak insufficient, medium sufficient)
      2 = strong model needed (neither weak nor medium sufficient)
    
    Returns:
        Labeling statistics dict.
    """
    if config is None:
        config = load_config()
    
    labeling_cfg = config.get("labeling", {})
    quality_threshold = labeling_cfg.get("quality_threshold", 7.0)
    score_field = labeling_cfg.get("score_field", "overall_score")
    label_map = labeling_cfg.get("label_map", {"weak": 0, "medium": 1, "strong": 2})
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Labeling: {input_path}")
    print(f"  Method: ultrafeedback_score")
    print(f"  Quality threshold: {quality_threshold}")
    print(f"  Score field: {score_field}")
    
    total = 0
    labeled = 0
    skipped_no_tier = 0
    skipped_no_score = 0
    label_counts = Counter()
    tier_coverage = Counter()  # How many models per tier we see
    
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
            
            # Group scores by tier
            tier_scores = {"weak": [], "medium": [], "strong": []}
            tier_models = {"weak": [], "medium": [], "strong": []}
            
            for comp in completions:
                model = comp.get("model", "")
                score = comp.get("overall_score")
                
                if score is None:
                    continue
                
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    continue
                
                tier = _get_model_tier(model, config)
                if tier and tier in tier_scores:
                    tier_scores[tier].append(score)
                    tier_models[tier].append(model)
                    tier_coverage[tier] += 1
            
            # Need at least some scored completions
            all_scores = tier_scores["weak"] + tier_scores["medium"] + tier_scores["strong"]
            if not all_scores:
                skipped_no_score += 1
                continue
            
            # Determine routing label
            # Logic: find the weakest tier that meets quality threshold
            best_weak = max(tier_scores["weak"]) if tier_scores["weak"] else 0
            best_medium = max(tier_scores["medium"]) if tier_scores["medium"] else 0
            best_strong = max(tier_scores["strong"]) if tier_scores["strong"] else 0
            
            if best_weak >= quality_threshold:
                routing_label = label_map["weak"]  # 0
            elif best_medium >= quality_threshold:
                routing_label = label_map["medium"]  # 1
            else:
                routing_label = label_map["strong"]  # 2
            
            # Create labeled record
            labeled_record = {
                "prompt": instruction,
                "routing_label": routing_label,
                "label_type": "derived_routing_label",
                "label_method": "ultrafeedback_score",
                "label_evidence": {
                    "best_weak_score": round(best_weak, 2),
                    "best_medium_score": round(best_medium, 2),
                    "best_strong_score": round(best_strong, 2),
                    "quality_threshold": quality_threshold,
                    "weak_models": tier_models["weak"],
                    "medium_models": tier_models["medium"],
                    "strong_models": tier_models["strong"],
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
        "method": "ultrafeedback_score",
        "quality_threshold": quality_threshold,
        "total_input": total,
        "total_labeled": labeled,
        "skipped_no_tier": skipped_no_tier,
        "skipped_no_score": skipped_no_score,
        "label_distribution": dict(label_counts),
        "tier_coverage": dict(tier_coverage),
    }
    
    print(f"\nLabeling complete!")
    print(f"  Input:          {total}")
    print(f"  Labeled:        {labeled}")
    print(f"  Skipped (tier): {skipped_no_tier}")
    print(f"  Skipped (score):{skipped_no_score}")
    print(f"\n  Label Distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / labeled * 100 if labeled > 0 else 0
        names = {0: "weak", 1: "medium", 2: "strong"}
        print(f"    {label} ({names.get(label, '?')}): {count} ({pct:.1f}%)")
    print(f"\n  Tier Coverage:")
    for tier, count in tier_coverage.items():
        print(f"    {tier}: {count} responses scored")
    
    # Save stats
    meta_path = Path(output_path).parent.parent / "metadata" / "labeling_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


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
    parser.add_argument("--method", type=str, default="ultrafeedback_score",
                        choices=["ultrafeedback_score", "mock_length", "all"])
    parser.add_argument("--input", type=str, default="data/processed/ultrafeedback_deduped.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/labeled")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    
    if args.method in ("ultrafeedback_score", "all"):
        label_ultrafeedback(args.input, f"{args.output_dir}/ultrafeedback_labeled.jsonl", config)
    
    if args.method in ("mock_length", "all"):
        label_mock_baseline(args.input, f"{args.output_dir}/mock_labeled.jsonl", config)
