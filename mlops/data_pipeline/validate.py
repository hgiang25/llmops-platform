"""
Schema Validation — Validate raw dataset schema and data quality.

Validates:
  - Required fields exist
  - Data types are correct
  - Values are within expected ranges
  - No critical missing data
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def validate_ultrafeedback(input_path: str = "data/raw/ultrafeedback_raw.jsonl") -> dict:
    """
    Validate schema and quality of ingested UltraFeedback data.
    
    Expected schema per row:
      - instruction: str (non-empty)
      - completions: list[dict] (non-empty, each with model, output, scores)
      - source: str
      - n_completions: int
    
    Returns:
        Validation report dict.
    """
    path = Path(input_path)
    if not path.exists():
        return {"error": f"File not found: {input_path}"}
    
    print(f"Validating: {input_path}")
    
    total = 0
    valid = 0
    errors = []
    warnings = []
    
    # Statistics
    missing_instruction = 0
    missing_completions = 0
    empty_completions = 0
    missing_scores = 0
    score_ranges = {"min": float("inf"), "max": float("-inf")}
    models_seen = set()
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total += 1
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Invalid JSON")
                continue
            
            is_valid = True
            
            # Check instruction
            instruction = record.get("instruction", "")
            if not instruction or not isinstance(instruction, str):
                missing_instruction += 1
                is_valid = False
            
            # Check completions
            completions = record.get("completions", [])
            if not completions:
                missing_completions += 1
                is_valid = False
            elif not isinstance(completions, list):
                errors.append(f"Line {line_num}: completions is not a list")
                is_valid = False
            else:
                if len(completions) == 0:
                    empty_completions += 1
                    is_valid = False
                
                for comp in completions:
                    model = comp.get("model", "")
                    if model:
                        models_seen.add(model)
                    
                    overall_score = comp.get("overall_score")
                    if overall_score is None:
                        missing_scores += 1
                    else:
                        try:
                            s = float(overall_score)
                            score_ranges["min"] = min(score_ranges["min"], s)
                            score_ranges["max"] = max(score_ranges["max"], s)
                        except (ValueError, TypeError):
                            warnings.append(f"Line {line_num}: Invalid score '{overall_score}'")
            
            if is_valid:
                valid += 1
    
    # Handle infinity
    if score_ranges["min"] == float("inf"):
        score_ranges = {"min": None, "max": None}
    
    report = {
        "file": input_path,
        "total_records": total,
        "valid_records": valid,
        "invalid_records": total - valid,
        "validity_rate": round(valid / total, 4) if total > 0 else 0,
        "missing_instruction": missing_instruction,
        "missing_completions": missing_completions,
        "empty_completions": empty_completions,
        "missing_scores": missing_scores,
        "score_range": score_ranges,
        "unique_models": sorted(list(models_seen)),
        "n_unique_models": len(models_seen),
        "errors": errors[:20],  # Cap at 20
        "warnings": warnings[:20],
    }
    
    # Print report
    print(f"\n{'=' * 50}")
    print("VALIDATION REPORT: UltraFeedback")
    print(f"{'=' * 50}")
    print(f"  Total records:       {total}")
    print(f"  Valid records:       {valid} ({report['validity_rate'] * 100:.1f}%)")
    print(f"  Missing instruction: {missing_instruction}")
    print(f"  Missing completions: {missing_completions}")
    print(f"  Missing scores:      {missing_scores}")
    print(f"  Score range:         {score_ranges}")
    print(f"  Unique models:       {len(models_seen)}")
    for m in sorted(models_seen):
        print(f"    - {m}")
    if errors:
        print(f"\n  First {min(len(errors), 5)} errors:")
        for e in errors[:5]:
            print(f"    {e}")
    print(f"{'=' * 50}")
    
    return report


def validate_chatbot_arena(input_path: str = "data/raw/chatbot_arena_raw.jsonl") -> dict:
    """Validate schema and quality of ingested Chatbot Arena data."""
    path = Path(input_path)
    if not path.exists():
        return {"error": f"File not found: {input_path}"}
    
    print(f"Validating: {input_path}")
    
    total = 0
    valid = 0
    missing_prompt = 0
    missing_models = 0
    missing_winner = 0
    models_seen = set()
    winner_dist = {"model_a": 0, "model_b": 0, "tie": 0}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            record = json.loads(line)
            is_valid = True
            
            if not record.get("prompt"):
                missing_prompt += 1
                is_valid = False
            
            if not record.get("model_a") or not record.get("model_b"):
                missing_models += 1
                is_valid = False
            else:
                models_seen.add(record["model_a"])
                models_seen.add(record["model_b"])
            
            wa = record.get("winner_model_a", 0)
            wb = record.get("winner_model_b", 0)
            wt = record.get("winner_tie", 0)
            
            if wa + wb + wt == 0:
                missing_winner += 1
                is_valid = False
            else:
                if wa:
                    winner_dist["model_a"] += 1
                elif wb:
                    winner_dist["model_b"] += 1
                else:
                    winner_dist["tie"] += 1
            
            if is_valid:
                valid += 1
    
    report = {
        "file": input_path,
        "total_records": total,
        "valid_records": valid,
        "invalid_records": total - valid,
        "validity_rate": round(valid / total, 4) if total > 0 else 0,
        "missing_prompt": missing_prompt,
        "missing_models": missing_models,
        "missing_winner": missing_winner,
        "winner_distribution": winner_dist,
        "unique_models": len(models_seen),
    }
    
    print(f"\n{'=' * 50}")
    print("VALIDATION REPORT: Chatbot Arena")
    print(f"{'=' * 50}")
    print(f"  Total records:  {total}")
    print(f"  Valid records:  {valid} ({report['validity_rate'] * 100:.1f}%)")
    print(f"  Winner dist:    {winner_dist}")
    print(f"  Unique models:  {len(models_seen)}")
    print(f"{'=' * 50}")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset schema")
    parser.add_argument("--source", type=str, default="ultrafeedback",
                        choices=["ultrafeedback", "chatbot_arena", "all"])
    parser.add_argument("--input_dir", type=str, default="data/raw")
    
    args = parser.parse_args()
    
    if args.source in ("ultrafeedback", "all"):
        validate_ultrafeedback(f"{args.input_dir}/ultrafeedback_raw.jsonl")
    
    if args.source in ("chatbot_arena", "all"):
        validate_chatbot_arena(f"{args.input_dir}/chatbot_arena_raw.jsonl")
