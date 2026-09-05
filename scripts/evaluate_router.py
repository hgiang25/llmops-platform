"""
Evaluate Router — Step 7: Full evaluation with baselines and routing metrics.

Usage:
    python scripts/evaluate_router.py
    python scripts/evaluate_router.py --model_path models/cloudops-llm-adapter
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from pathlib import Path

from mlops.evaluation.evaluator import ModelEvaluator
from mlops.evaluation.baselines import run_all_baselines
from mlops.evaluation.routing_metrics import compute_routing_metrics, print_routing_report


def load_test_data(test_path: str = "data/splits/test.jsonl") -> list[dict]:
    """Load test data from JSONL."""
    if not Path(test_path).exists():
        print(f"ERROR: Test data not found at {test_path}")
        return []
    
    data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Router")
    parser.add_argument("--model_path", type=str, default="models/cloudops-llm-adapter",
                        help="Path to trained model/adapter")
    parser.add_argument("--test_path", type=str, default="data/splits/test.jsonl",
                        help="Path to test data")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model inference (only run baselines)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  STEP 7: EVALUATE ROUTER")
    print("=" * 70)
    
    # Load test data
    test_data = load_test_data(args.test_path)
    if not test_data:
        return
    
    print(f"\n  Test samples: {len(test_data)}")
    
    true_labels = [r.get("routing_label", 0) for r in test_data]
    
    evaluator = ModelEvaluator()
    all_results = {}
    
    # --- Baselines ---
    print("\n" + "=" * 60)
    print("  BASELINE EVALUATION")
    print("=" * 60)
    
    baseline_results = run_all_baselines(test_data)
    
    for baseline_name, baseline_data in baseline_results.items():
        preds = baseline_data["predictions"]
        
        # Classification metrics
        report = evaluator.evaluate(
            test_data,
            predicted_classes=preds,
            model_name=baseline_name,
        )
        
        # Routing metrics
        routing = compute_routing_metrics(true_labels, preds)
        
        all_results[baseline_name] = {
            "classification": report,
            "routing": routing,
        }
        
        print(f"\n--- {baseline_name} ---")
        print(f"  Accuracy:  {report.get('routing_accuracy', 'N/A')}")
        print(f"  F1 Macro:  {report.get('routing_f1_macro', 'N/A')}")
        print(f"  F1 Weight: {report.get('routing_f1', 'N/A')}")
        print(f"  Per-class: {report.get('per_class_f1', {})}")
        print(f"  Cost savings: {routing.get('cost_savings_vs_always_strong', 'N/A')}")
        print(f"  Weak failures: {routing.get('weak_failure_rate', 'N/A')}")
    
    # --- Fine-tuned Model ---
    if not args.skip_model:
        model_path = args.model_path
        if not Path(model_path).exists():
            print(f"\n  WARNING: Model not found at {model_path}. Skipping model evaluation.")
            print("  Train a model first: python scripts/train_router.py")
        else:
            print(f"\n{'=' * 60}")
            print(f"  FINE-TUNED MODEL EVALUATION: {model_path}")
            print(f"{'=' * 60}")
            
            # Run inference
            predicted_classes = evaluator.predict_classes(model_path, test_data)
            
            # Classification metrics
            model_report = evaluator.evaluate(
                test_data,
                predicted_classes=predicted_classes,
                model_name="finetuned_router",
            )
            
            # Routing metrics
            model_routing = compute_routing_metrics(true_labels, predicted_classes)
            print_routing_report(model_routing)
            
            all_results["finetuned_router"] = {
                "classification": model_report,
                "routing": model_routing,
            }
            
            print(f"\n  Accuracy:  {model_report.get('routing_accuracy', 'N/A')}")
            print(f"  F1 Macro:  {model_report.get('routing_f1_macro', 'N/A')}")
            print(f"  F1 Weight: {model_report.get('routing_f1', 'N/A')}")
            print(f"  Per-class: {model_report.get('per_class_f1', {})}")
    
    # --- Summary Table ---
    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weight':>10} {'Cost Save':>10} {'Weak Fail':>10}")
    print(f"  {'-' * 75}")
    
    for name, data in all_results.items():
        cls = data.get("classification", {})
        rtg = data.get("routing", {})
        print(f"  {name:<25} "
              f"{cls.get('routing_accuracy', 0):>10.4f} "
              f"{cls.get('routing_f1_macro', 0):>10.4f} "
              f"{cls.get('routing_f1', 0):>10.4f} "
              f"{rtg.get('cost_savings_vs_always_strong', 0):>10.4f} "
              f"{rtg.get('weak_failure_rate', 0):>10.4f}")
    
    print(f"{'=' * 70}")
    
    # Save results
    result_path = Path("reports/evaluation") / "evaluation_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable
    serializable = {}
    for name, data in all_results.items():
        serializable[name] = {
            "classification": {k: v for k, v in data.get("classification", {}).items()
                             if k != "confusion_matrix" or isinstance(v, (dict, list, str, int, float))},
            "routing": data.get("routing", {}),
        }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
