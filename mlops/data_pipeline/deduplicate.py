"""
Deduplication — Remove exact and near-duplicate prompts.

Critical for research integrity:
  - Prevents data leakage between train/val/test
  - Removes redundant training signals
  - Ensures diversity in training data

Methods:
  - Exact match: hash-based deduplication
  - Near-duplicate: Jaccard similarity on word n-grams
"""

import json
import hashlib
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import yaml


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison."""
    return text.lower().strip()


def _hash_text(text: str) -> str:
    """SHA-256 hash of normalized text."""
    normalized = _normalize_for_dedup(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_ngrams(text: str, n: int = 3) -> set:
    """Extract word n-grams from text."""
    words = _normalize_for_dedup(text).split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate(
    input_path: str = "data/processed/ultrafeedback_cleaned.jsonl",
    output_path: str = "data/processed/ultrafeedback_deduped.jsonl",
    config: Optional[dict] = None,
    prompt_field: str = "instruction",
) -> dict:
    """
    Remove exact and near-duplicate prompts.
    
    Args:
        input_path: Path to cleaned JSONL.
        output_path: Path to output JSONL.
        config: Dataset config dict.
        prompt_field: Field name for the prompt text.
    
    Returns:
        Deduplication statistics dict.
    """
    if config is None:
        config = load_config()
    
    dedup_cfg = config.get("deduplication", {})
    do_exact = dedup_cfg.get("exact_match", True)
    do_near = dedup_cfg.get("near_duplicate", True)
    near_threshold = dedup_cfg.get("near_duplicate_threshold", 0.85)
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Deduplicating: {input_path}")
    print(f"  Exact match: {do_exact}")
    print(f"  Near-duplicate: {do_near} (threshold={near_threshold})")
    
    # Phase 1: Load all records and compute hashes
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    total = len(records)
    print(f"  Total records: {total}")
    
    # Phase 2: Exact dedup
    exact_dupes = 0
    seen_hashes = set()
    unique_records = []
    
    if do_exact:
        for record in records:
            prompt = record.get(prompt_field, "")
            h = _hash_text(prompt)
            if h in seen_hashes:
                exact_dupes += 1
            else:
                seen_hashes.add(h)
                unique_records.append(record)
        print(f"  Exact duplicates removed: {exact_dupes}")
    else:
        unique_records = records
    
    # Phase 3: Near-duplicate detection
    near_dupes = 0
    
    if do_near and len(unique_records) > 0:
        # For large datasets, use MinHash or approximate methods.
        # For our scale (~60K), brute-force on a sample is feasible.
        # We use a bucket-based approach: group by similar length, 
        # then compare within buckets.
        
        final_records = []
        kept_ngrams = []
        
        # Sort by length for efficiency
        unique_records.sort(key=lambda r: len(r.get(prompt_field, "")))
        
        for i, record in enumerate(unique_records):
            prompt = record.get(prompt_field, "")
            ngrams = _word_ngrams(prompt)
            
            is_near_dupe = False
            
            # Only compare with recent records in similar length range
            # to avoid O(n²) for entire dataset
            compare_window = min(100, len(kept_ngrams))
            for j in range(len(kept_ngrams) - 1, max(-1, len(kept_ngrams) - compare_window - 1), -1):
                sim = _jaccard_similarity(ngrams, kept_ngrams[j])
                if sim >= near_threshold:
                    is_near_dupe = True
                    break
            
            if is_near_dupe:
                near_dupes += 1
            else:
                final_records.append(record)
                kept_ngrams.append(ngrams)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(unique_records)} for near-dedup...")
        
        unique_records = final_records
        print(f"  Near-duplicates removed: {near_dupes}")
    
    # Save results
    with open(out_path, "w", encoding="utf-8") as f:
        for record in unique_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "total_input": total,
        "exact_duplicates": exact_dupes,
        "near_duplicates": near_dupes,
        "total_output": len(unique_records),
        "dedup_rate": round(1 - len(unique_records) / total, 4) if total > 0 else 0,
    }
    
    print(f"\nDeduplication complete!")
    print(f"  Input:    {total}")
    print(f"  Output:   {len(unique_records)}")
    print(f"  Removed:  {exact_dupes + near_dupes} ({stats['dedup_rate'] * 100:.1f}%)")
    
    # Save stats
    meta_path = Path(output_path).parent.parent / "metadata" / "dedup_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate dataset")
    parser.add_argument("--input", type=str, default="data/processed/ultrafeedback_cleaned.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/ultrafeedback_deduped.jsonl")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    deduplicate(args.input, args.output, config)
