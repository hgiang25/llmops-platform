"""
Sample Dataset — Step 4: Balanced sampling for training.

Usage:
    python scripts/sample_dataset.py
    python scripts/sample_dataset.py --max_samples 3000 --strategy balanced
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from mlops.data_pipeline.sample import sample_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample dataset")
    parser.add_argument("--input", type=str, default="data/labeled/ultrafeedback_labeled.jsonl")
    parser.add_argument("--output", type=str, default="data/labeled/ultrafeedback_sampled.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["random", "stratified", "balanced"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  STEP 4: SAMPLE DATASET")
    print("=" * 70)
    
    sample_dataset(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        strategy=args.strategy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
