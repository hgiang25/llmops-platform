"""
Train Router — Step 6: Fine-tune router model.

Supports two architectures:
  - DeBERTa-v3-base + Ordinal Classification (recommended)
  - Qwen2.5-0.5B + QLoRA Standard Classification (baseline)

Usage:
    python scripts/train_router.py
    python scripts/train_router.py --architecture ordinal_classification
    python scripts/train_router.py --architecture standard_classification
    python scripts/train_router.py --mock
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train LLM Router")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Training config YAML")
    parser.add_argument("--architecture", type=str, default=None,
                        choices=["ordinal_classification", "standard_classification"],
                        help="Model architecture (default: from config)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override train dataset path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock training (no GPU)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path("mlops/training/train_config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Determine architecture
    architecture = args.architecture or config.get("model", {}).get("architecture", "ordinal_classification")
    
    print("=" * 70)
    print(f"  STEP 6: TRAIN ROUTER ({architecture})")
    print("=" * 70)
    
    # Override from args
    dataset_path = args.dataset or config.get("dataset", {}).get("train_path", "data/splits/train.jsonl")
    output_dir = args.output_dir or config.get("training", {}).get("output_dir", "models/deberta-router")
    
    if not Path(dataset_path).exists():
        print(f"\nERROR: Training data not found at {dataset_path}")
        print("Run the data pipeline first:")
        print("  python scripts/download_dataset.py")
        print("  python scripts/build_dataset.py")
        print("  python scripts/create_labels.py")
        print("  python scripts/sample_dataset.py")
        print("  python scripts/split_dataset.py")
        return
    
    # Count samples
    n_samples = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        n_samples = sum(1 for line in f if line.strip())
    
    print(f"\n  Config:       {config_path}")
    print(f"  Architecture: {architecture}")
    print(f"  Dataset:      {dataset_path} ({n_samples} samples)")
    print(f"  Output:       {output_dir}")
    print(f"  Model:        {config.get('model', {}).get('base_model', 'unknown')}")
    print(f"  Mock:         {args.mock}")
    
    from mlops.training.finetune import DeBERTaOrdinalTrainer, QLoRATrainer
    
    if architecture == "ordinal_classification":
        trainer = DeBERTaOrdinalTrainer(config=config)
    else:
        trainer = QLoRATrainer(config=config)
    
    result = trainer.train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        mock=args.mock if args.mock else None,
    )
    
    print(f"\n{'=' * 70}")
    print("  TRAINING RESULT")
    print(f"{'=' * 70}")
    print(json.dumps(result, indent=2))
    
    # Save result
    result_path = Path("reports/experiments") / "latest_training_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {result_path}")


if __name__ == "__main__":
    main()
