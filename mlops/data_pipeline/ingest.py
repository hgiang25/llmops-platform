"""
Dataset Ingestion — Download and ingest datasets from HuggingFace.

Supports:
  - openbmb/UltraFeedback: GPT-4 annotated multi-model completions
  - lmsys/lmsys-arena-human-preference-55k: Human preference pairwise battles

Design:
  - Streaming mode available for large datasets
  - Saves raw data in JSONL format for downstream processing
  - Resource-aware: does NOT load entire dataset into RAM by default
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_path: str = None) -> dict:
    """Load dataset configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest_ultrafeedback(
    output_dir: str = "data/raw",
    max_samples: int = None,
    streaming: bool = False,
) -> dict:
    """
    Download and save UltraFeedback dataset.
    
    Schema (per row):
      - instruction: str (the prompt)
      - completions: list of dicts, each with:
          - model: str (model name)
          - output: str (response text)
          - annotations: dict with scores
    
    Returns:
        Dict with ingestion statistics.
    """
    # pyrefly: ignore [missing-import]
    from datasets import load_dataset
    
    print("=" * 60)
    print("INGESTING: openbmb/UltraFeedback")
    print("=" * 60)
    
    output_path = Path(output_dir) / "ultrafeedback_raw.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Load dataset
    print("Loading dataset from HuggingFace...")
    if streaming:
        dataset = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    else:
        dataset = load_dataset("openbmb/UltraFeedback", split="train")
    
    count = 0
    skipped = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            # Extract and validate core fields
            instruction = row.get("instruction", "")
            completions = row.get("completions", [])
            source = row.get("source", "")
            
            if not instruction or not completions:
                skipped += 1
                continue
            
            # Normalize completions to extract scores
            normalized_completions = []
            for comp in completions:
                model = comp.get("model", "unknown")
                output = comp.get("output", "")
                annotations = comp.get("annotations", {})
                
                # Extract scores from annotations
                scores = {}
                if annotations:
                    for aspect_name, aspect_data in annotations.items():
                        if isinstance(aspect_data, dict):
                            rating = aspect_data.get("Rating", aspect_data.get("rating", ""))
                            # Parse rating (can be "4", "4.0", etc.)
                            try:
                                scores[aspect_name] = float(str(rating).strip())
                            except (ValueError, TypeError):
                                pass
                
                # Calculate overall score
                overall_score = None
                if "overall_score" in annotations:
                    try:
                        if isinstance(annotations["overall_score"], dict):
                            overall_score = float(str(annotations["overall_score"].get("Rating", 0)).strip())
                        else:
                            overall_score = float(str(annotations["overall_score"]).strip())
                    except (ValueError, TypeError):
                        pass
                
                if overall_score is None and scores:
                    overall_score = sum(scores.values()) / len(scores)
                
                normalized_completions.append({
                    "model": model,
                    "output": output[:500],  # Truncate response to save space
                    "scores": scores,
                    "overall_score": overall_score,
                })
            
            record = {
                "instruction": instruction,
                "completions": normalized_completions,
                "source": source,
                "n_completions": len(normalized_completions),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 5000 == 0:
                print(f"  Ingested {count} records...")
            
            if max_samples and count >= max_samples:
                break
    
    elapsed = time.time() - start_time
    
    stats = {
        "dataset": "openbmb/UltraFeedback",
        "output_path": str(output_path),
        "total_ingested": count,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    print(f"\nIngestion complete!")
    print(f"  Records: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    
    # Save metadata
    meta_path = Path(output_dir).parent / "metadata" / "ingestion_ultrafeedback.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def ingest_chatbot_arena(
    output_dir: str = "data/raw",
    max_samples: int = None,
) -> dict:
    """
    Download and save Chatbot Arena dataset.
    
    Schema (per row):
      - prompt: str
      - response_a: str
      - response_b: str
      - model_a: str
      - model_b: str
      - winner_model_a: int (0 or 1)
      - winner_model_b: int (0 or 1)
      - winner_tie: int (0 or 1)
    
    Returns:
        Dict with ingestion statistics.
    """
    # pyrefly: ignore [missing-import]
    from datasets import load_dataset
    
    print("=" * 60)
    print("INGESTING: lmsys/lmsys-arena-human-preference-55k")
    print("=" * 60)
    
    output_path = Path(output_dir) / "chatbot_arena_raw.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        dataset = load_dataset(
            "lmsys/lmsys-arena-human-preference-55k",
            split="train",
        )
    except Exception as e:
        print(f"\n[WARNING] Could not load Chatbot Arena dataset: {e}")
        print("This dataset may require accepting terms on HuggingFace.")
        print("Falling back to UltraFeedback as primary dataset.")
        return {"error": str(e), "dataset": "lmsys/lmsys-arena-human-preference-55k"}
    
    count = 0
    skipped = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            prompt = row.get("prompt", "")
            if not prompt:
                skipped += 1
                continue
            
            record = {
                "prompt": prompt,
                "response_a": (row.get("response_a", "") or "")[:500],
                "response_b": (row.get("response_b", "") or "")[:500],
                "model_a": row.get("model_a", ""),
                "model_b": row.get("model_b", ""),
                "winner_model_a": row.get("winner_model_a", 0),
                "winner_model_b": row.get("winner_model_b", 0),
                "winner_tie": row.get("winner_tie", 0),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 5000 == 0:
                print(f"  Ingested {count} records...")
            
            if max_samples and count >= max_samples:
                break
    
    elapsed = time.time() - start_time
    
    stats = {
        "dataset": "lmsys/lmsys-arena-human-preference-55k",
        "output_path": str(output_path),
        "total_ingested": count,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    print(f"\nIngestion complete!")
    print(f"  Records: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    
    meta_path = Path(output_dir).parent / "metadata" / "ingestion_chatbot_arena.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Ingestion for LLM Router")
    parser.add_argument("--source", type=str, default="ultrafeedback",
                        choices=["ultrafeedback", "chatbot_arena", "all"],
                        help="Dataset source to ingest")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to ingest (None = all)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")
    
    args = parser.parse_args()
    
    if args.source in ("ultrafeedback", "all"):
        ingest_ultrafeedback(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            streaming=args.streaming,
        )
    
    if args.source in ("chatbot_arena", "all"):
        ingest_chatbot_arena(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
