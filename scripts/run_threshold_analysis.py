"""
Run Threshold Analysis — Analyze label distribution across different thresholds.

This script helps researchers select the appropriate quality threshold (τ)
for Minimum Sufficient Model labeling by showing how the distribution of
weak/medium/strong labels changes for different τ values.

Usage:
    python scripts/run_threshold_analysis.py
    python scripts/run_threshold_analysis.py --max_samples 1000
    python scripts/run_threshold_analysis.py --thresholds 3.5 4.0 4.5
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from mlops.data_pipeline.threshold_analysis import run_threshold_analysis


def main():
    parser = argparse.ArgumentParser(description="Run threshold sensitivity analysis")
    parser.add_argument("--input", type=str, default="data/processed/ultrafeedback_deduped.jsonl")
    parser.add_argument("--output", type=str, default="data/metadata/threshold_analysis.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for faster analysis")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="List of thresholds to test (e.g., 3.5 4.0 4.5)")
    parser.add_argument("--margins", type=float, nargs="+", default=None,
                        help="List of margins to test (e.g., 0.5 0.75 1.0)")
    
    args = parser.parse_args()
    
    run_threshold_analysis(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        thresholds=args.thresholds,
        margins=args.margins,
    )


if __name__ == "__main__":
    main()
