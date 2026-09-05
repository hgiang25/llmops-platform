"""
Dataset Sampling — Stratified and balanced sampling for training.

Ensures:
  - Configurable dataset size (for hardware constraints)
  - Class-balanced sampling
  - Reproducible (seeded)
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

import yaml


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_dataset(
    input_path: str = "data/labeled/ultrafeedback_labeled.jsonl",
    output_path: str = "data/labeled/ultrafeedback_sampled.jsonl",
    max_samples: int = None,
    strategy: str = None,
    seed: int = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Sample a labeled dataset with optional class balancing.
    
    Strategies:
      - "random": Simple random sampling
      - "stratified": Maintain original class proportions
      - "balanced": Equal samples per class (undersample majority)
    
    Returns:
        Sampling statistics dict.
    """
    if config is None:
        config = load_config()
    
    sampling_cfg = config.get("sampling", {})
    max_samples = max_samples or sampling_cfg.get("max_samples", 5000)
    strategy = strategy or sampling_cfg.get("strategy", "stratified")
    seed = seed if seed is not None else sampling_cfg.get("random_seed", 42)
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Sampling: {input_path}")
    print(f"  Strategy: {strategy}")
    print(f"  Max samples: {max_samples}")
    print(f"  Seed: {seed}")
    
    rng = random.Random(seed)
    
    # Load all records grouped by label
    records_by_label = defaultdict(list)
    total = 0
    
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get("routing_label", 0)
            records_by_label[label].append(record)
            total += 1
    
    before_dist = {k: len(v) for k, v in sorted(records_by_label.items())}
    print(f"\n  Before sampling:")
    for label, count in before_dist.items():
        names = {0: "weak", 1: "medium", 2: "strong"}
        print(f"    Class {label} ({names.get(label, '?')}): {count}")
    
    # If total is already <= max_samples, keep all
    if total <= max_samples:
        print(f"\n  Dataset ({total}) <= max_samples ({max_samples}). Keeping all.")
        sampled = []
        for records in records_by_label.values():
            sampled.extend(records)
        rng.shuffle(sampled)
    
    elif strategy == "random":
        all_records = []
        for records in records_by_label.values():
            all_records.extend(records)
        sampled = rng.sample(all_records, min(max_samples, len(all_records)))
    
    elif strategy == "stratified":
        # Maintain original proportions
        sampled = []
        for label in sorted(records_by_label.keys()):
            records = records_by_label[label]
            proportion = len(records) / total
            n_take = max(1, int(max_samples * proportion))
            n_take = min(n_take, len(records))
            sampled.extend(rng.sample(records, n_take))
        rng.shuffle(sampled)
    
    elif strategy == "balanced":
        # Equal samples per class
        n_classes = len(records_by_label)
        per_class = max(1, max_samples // n_classes)
        sampled = []
        for label in sorted(records_by_label.keys()):
            records = records_by_label[label]
            n_take = min(per_class, len(records))
            sampled.extend(rng.sample(records, n_take))
        rng.shuffle(sampled)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        for record in sampled:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Compute after distribution
    after_dist = Counter()
    for record in sampled:
        after_dist[record.get("routing_label", 0)] += 1
    
    print(f"\n  After sampling:")
    for label, count in sorted(after_dist.items()):
        names = {0: "weak", 1: "medium", 2: "strong"}
        print(f"    Class {label} ({names.get(label, '?')}): {count}")
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "strategy": strategy,
        "max_samples": max_samples,
        "seed": seed,
        "total_input": total,
        "total_output": len(sampled),
        "before_distribution": before_dist,
        "after_distribution": dict(after_dist),
    }
    
    # Save stats
    meta_path = Path(output_path).parent.parent / "metadata" / "sampling_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataset")
    parser.add_argument("--input", type=str, default="data/labeled/ultrafeedback_labeled.jsonl")
    parser.add_argument("--output", type=str, default="data/labeled/ultrafeedback_sampled.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["random", "stratified", "balanced"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    sample_dataset(args.input, args.output, args.max_samples, args.strategy, args.seed, config)
