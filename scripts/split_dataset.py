"""
Split Dataset — Step 5: Create train/val/test splits.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from mlops.data_pipeline.split import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", type=str, default="data/labeled/ultrafeedback_sampled.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  STEP 5: SPLIT DATASET")
    print("=" * 70)
    
    split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
