"""
Data Cleaning — Clean and normalize raw dataset records.

Operations:
  - Remove empty/invalid prompts
  - Normalize whitespace
  - Filter by prompt length
  - Remove records with missing scores
  - Standardize field names
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace and strip."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_ultrafeedback(
    input_path: str = "data/raw/ultrafeedback_raw.jsonl",
    output_path: str = "data/processed/ultrafeedback_cleaned.jsonl",
    config: Optional[dict] = None,
) -> dict:
    """
    Clean UltraFeedback raw data.
    
    Returns:
        Cleaning statistics dict.
    """
    if config is None:
        config = load_config()
    
    cleaning_cfg = config.get("cleaning", {})
    min_length = cleaning_cfg.get("min_prompt_length", 10)
    max_length = cleaning_cfg.get("max_prompt_length", 4096)
    min_words = cleaning_cfg.get("min_prompt_words", 3)
    do_normalize = cleaning_cfg.get("normalize_whitespace", True)
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}
    
    print(f"Cleaning: {input_path}")
    
    total = 0
    kept = 0
    removed_empty = 0
    removed_short = 0
    removed_long = 0
    removed_few_words = 0
    removed_no_scores = 0
    
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            record = json.loads(line)
            
            instruction = record.get("instruction", "")
            
            # Normalize whitespace
            if do_normalize:
                instruction = normalize_whitespace(instruction)
                record["instruction"] = instruction
            
            # Remove empty
            if not instruction:
                removed_empty += 1
                continue
            
            # Length filter
            if len(instruction) < min_length:
                removed_short += 1
                continue
            
            if len(instruction) > max_length:
                removed_long += 1
                continue
            
            # Word count filter
            if len(instruction.split()) < min_words:
                removed_few_words += 1
                continue
            
            # Check that at least one completion has a valid score
            completions = record.get("completions", [])
            has_valid_score = any(
                c.get("overall_score") is not None
                for c in completions
            )
            
            if not has_valid_score:
                removed_no_scores += 1
                continue
            
            # Passed all filters
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "total_input": total,
        "total_output": kept,
        "removed_empty": removed_empty,
        "removed_short": removed_short,
        "removed_long": removed_long,
        "removed_few_words": removed_few_words,
        "removed_no_scores": removed_no_scores,
        "removal_rate": round(1 - kept / total, 4) if total > 0 else 0,
    }
    
    print(f"\nCleaning complete!")
    print(f"  Input:           {total}")
    print(f"  Output:          {kept}")
    print(f"  Removed empty:   {removed_empty}")
    print(f"  Removed short:   {removed_short}")
    print(f"  Removed long:    {removed_long}")
    print(f"  Removed no words:{removed_few_words}")
    print(f"  Removed no score:{removed_no_scores}")
    print(f"  Removal rate:    {stats['removal_rate'] * 100:.1f}%")
    
    # Save stats
    meta_path = Path(output_path).parent.parent / "metadata" / "cleaning_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset")
    parser.add_argument("--input", type=str, default="data/raw/ultrafeedback_raw.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/ultrafeedback_cleaned.jsonl")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    clean_ultrafeedback(args.input, args.output, config)
