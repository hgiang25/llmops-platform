"""
Download Dataset — Step 1 of the data pipeline.

Downloads UltraFeedback (and optionally Chatbot Arena) from HuggingFace.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --source ultrafeedback
    python scripts/download_dataset.py --source all --max_samples 10000
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from mlops.data_pipeline.ingest import ingest_ultrafeedback, ingest_chatbot_arena


def main():
    parser = argparse.ArgumentParser(description="Download datasets for LLM Router")
    parser.add_argument("--source", type=str, default="ultrafeedback",
                        choices=["ultrafeedback", "chatbot_arena", "all"])
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples (None = all)")
    parser.add_argument("--streaming", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  STEP 1: DOWNLOAD DATASET")
    print("=" * 70)
    
    if args.source in ("ultrafeedback", "all"):
        result = ingest_ultrafeedback(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            streaming=args.streaming,
        )
        if "error" in result:
            print(f"ERROR: {result['error']}")
    
    if args.source in ("chatbot_arena", "all"):
        result = ingest_chatbot_arena(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
        if "error" in result:
            print(f"WARNING: {result['error']}")
            print("Chatbot Arena is optional. Continuing with UltraFeedback only.")


if __name__ == "__main__":
    main()
