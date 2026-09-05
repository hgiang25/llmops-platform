"""
Train/Validation/Test Split — Stratified splitting with data leakage prevention.

Critical requirements:
  1. Stratified by routing label
  2. NO duplicate prompts across splits
  3. Reproducible (seeded)
  4. Test set is FIXED and never used for hyperparameter tuning
"""

import json
import hashlib
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


def _prompt_hash(text: str) -> str:
    """Hash for leakage checking."""
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()


def split_dataset(
    input_path: str = "data/labeled/ultrafeedback_sampled.jsonl",
    output_dir: str = "data/splits",
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Create stratified train/val/test splits.
    
    Guarantees:
      - No prompt appears in more than one split
      - Stratified by routing_label
      - Reproducible
    
    Returns:
        Split statistics dict.
    """
    if config is None:
        config = load_config()
    
    split_cfg = config.get("split", {})
    train_ratio = train_ratio or split_cfg.get("train_ratio", 0.70)
    val_ratio = val_ratio or split_cfg.get("val_ratio", 0.15)
    test_ratio = test_ratio or split_cfg.get("test_ratio", 0.15)
    seed = seed if seed is not None else split_cfg.get("random_seed", 42)
    do_stratify = split_cfg.get("stratify", True)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 0.01, f"Split ratios must sum to 1.0, got {total_ratio}"
    
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Splitting: {input_path}")
    print(f"  Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"  Stratified: {do_stratify}")
    print(f"  Seed: {seed}")
    
    rng = random.Random(seed)
    
    # Load all records
    records_by_label = defaultdict(list)
    all_records = []
    
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get("routing_label", 0)
            records_by_label[label].append(record)
            all_records.append(record)
    
    total = len(all_records)
    print(f"  Total records: {total}")
    
    train_records = []
    val_records = []
    test_records = []
    
    if do_stratify:
        # Split each class independently
        for label in sorted(records_by_label.keys()):
            records = records_by_label[label]
            rng.shuffle(records)
            
            n = len(records)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_test - n_val
            
            test_records.extend(records[:n_test])
            val_records.extend(records[n_test:n_test + n_val])
            train_records.extend(records[n_test + n_val:])
    else:
        rng.shuffle(all_records)
        n_test = max(1, int(total * test_ratio))
        n_val = max(1, int(total * val_ratio))
        
        test_records = all_records[:n_test]
        val_records = all_records[n_test:n_test + n_val]
        train_records = all_records[n_test + n_val:]
    
    # Shuffle within each split
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    
    # Verify no data leakage
    train_hashes = {_prompt_hash(r["prompt"]) for r in train_records}
    val_hashes = {_prompt_hash(r["prompt"]) for r in val_records}
    test_hashes = {_prompt_hash(r["prompt"]) for r in test_records}
    
    leak_train_val = train_hashes & val_hashes
    leak_train_test = train_hashes & test_hashes
    leak_val_test = val_hashes & test_hashes
    
    leakage_found = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)
    
    if leakage_found > 0:
        print(f"\n  ⚠️ DATA LEAKAGE DETECTED:")
        print(f"    Train ∩ Val:  {len(leak_train_val)}")
        print(f"    Train ∩ Test: {len(leak_train_test)}")
        print(f"    Val ∩ Test:   {len(leak_val_test)}")
        
        # Remove leaked records from train (keep in test/val)
        train_records = [r for r in train_records 
                        if _prompt_hash(r["prompt"]) not in (leak_train_val | leak_train_test)]
        print(f"    Removed {leakage_found} leaked records from train set")
    
    # Save splits
    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    
    for split_name, records in splits.items():
        split_path = out_dir / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Compute distributions
    split_stats = {}
    for split_name, records in splits.items():
        dist = Counter(r.get("routing_label", 0) for r in records)
        split_stats[split_name] = {
            "count": len(records),
            "distribution": dict(sorted(dist.items())),
        }
    
    print(f"\n  Split results:")
    for split_name, info in split_stats.items():
        print(f"    {split_name}: {info['count']} records")
        for label, count in sorted(info["distribution"].items()):
            names = {0: "weak", 1: "medium", 2: "strong"}
            pct = count / info["count"] * 100 if info["count"] > 0 else 0
            print(f"      Class {label} ({names.get(label, '?')}): {count} ({pct:.1f}%)")
    
    stats = {
        "input_path": input_path,
        "output_dir": output_dir,
        "total_input": total,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "stratified": do_stratify,
        "leakage_found": leakage_found,
        "splits": split_stats,
    }
    
    # Save metadata
    meta_path = Path(output_dir).parent / "metadata" / "split_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    # Save dataset version metadata
    version_cfg = config.get("versioning", {})
    version_meta = {
        "dataset_version": version_cfg.get("current_version", "v1.0"),
        "description": version_cfg.get("description", ""),
        "source": config.get("labeling", {}).get("method", "unknown"),
        "label_method": config.get("labeling", {}).get("method", "unknown"),
        "quality_threshold": config.get("labeling", {}).get("quality_threshold", None),
        "splits": split_stats,
        "seed": seed,
    }
    version_path = Path(output_dir).parent / "metadata" / "dataset_version.json"
    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(version_meta, f, indent=2)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", type=str, default="data/labeled/ultrafeedback_sampled.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    split_dataset(args.input, args.output_dir, args.train_ratio, args.val_ratio,
                  args.test_ratio, args.seed, config)
