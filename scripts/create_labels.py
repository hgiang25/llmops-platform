"""
Create Labels — Step 3: Construct routing labels from model scores.

Usage:
    python scripts/create_labels.py
    python scripts/create_labels.py --method all
    python scripts/create_labels.py --method mock_length
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from mlops.data_pipeline.label import label_ultrafeedback, label_mock_baseline


def main():
    parser = argparse.ArgumentParser(description="Construct routing labels")
    parser.add_argument("--method", type=str, default="ultrafeedback_score",
                        choices=["ultrafeedback_score", "mock_length", "all"])
    parser.add_argument("--input", type=str, default="data/processed/ultrafeedback_deduped.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/labeled")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  STEP 3: CREATE ROUTING LABELS")
    print("=" * 70)
    
    if args.method in ("ultrafeedback_score", "all"):
        print("\n--- Method: UltraFeedback Score-based ---")
        label_ultrafeedback(
            input_path=args.input,
            output_path=f"{args.output_dir}/ultrafeedback_labeled.jsonl",
        )
    
    if args.method in ("mock_length", "all"):
        print("\n--- Method: Mock Length Heuristic (BASELINE ONLY) ---")
        label_mock_baseline(
            input_path=args.input,
            output_path=f"{args.output_dir}/mock_labeled.jsonl",
        )


if __name__ == "__main__":
    main()
