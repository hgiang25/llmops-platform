"""
Build Dataset — Step 2: Validate, clean, and deduplicate raw data.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --input data/raw/ultrafeedback_raw.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from mlops.data_pipeline.validate import validate_ultrafeedback
from mlops.data_pipeline.clean import clean_ultrafeedback
from mlops.data_pipeline.deduplicate import deduplicate


def main():
    parser = argparse.ArgumentParser(description="Build dataset (validate, clean, deduplicate)")
    parser.add_argument("--input", type=str, default="data/raw/ultrafeedback_raw.jsonl")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  STEP 2: BUILD DATASET (Validate → Clean → Deduplicate)")
    print("=" * 70)
    
    # Step 2a: Validate
    print("\n--- Step 2a: Schema Validation ---")
    val_report = validate_ultrafeedback(args.input)
    if "error" in val_report:
        print(f"ERROR: {val_report['error']}")
        return
    
    # Step 2b: Clean
    print("\n--- Step 2b: Data Cleaning ---")
    clean_report = clean_ultrafeedback(
        input_path=args.input,
        output_path="data/processed/ultrafeedback_cleaned.jsonl",
    )
    if "error" in clean_report:
        print(f"ERROR: {clean_report['error']}")
        return
    
    # Step 2c: Deduplicate
    print("\n--- Step 2c: Deduplication ---")
    dedup_report = deduplicate(
        input_path="data/processed/ultrafeedback_cleaned.jsonl",
        output_path="data/processed/ultrafeedback_deduped.jsonl",
    )
    if "error" in dedup_report:
        print(f"ERROR: {dedup_report['error']}")
        return
    
    print("\n" + "=" * 70)
    print("  BUILD COMPLETE")
    print("=" * 70)
    print(f"  Raw:        {val_report.get('total_records', '?')} records")
    print(f"  Cleaned:    {clean_report.get('total_output', '?')} records")
    print(f"  Deduped:    {dedup_report.get('total_output', '?')} records")
    print(f"  Output:     data/processed/ultrafeedback_deduped.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    main()
