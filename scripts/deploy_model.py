"""
Deploy Model — Push manually trained model to MLflow.

Usage:
    python scripts/deploy_model.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlops.registry.mlflow_utils import ModelRegistry

def main():
    print("=" * 70)
    print("  DEPLOY MODEL TO MLFLOW")
    print("=" * 70)

    model_path = "models/cloudops-llm-adapter"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Run train_router.py first.")
        return

    # Load latest evaluation metrics if available
    metrics = {}
    eval_report = "reports/evaluation/evaluation_results.json"
    if os.path.exists(eval_report):
        with open(eval_report, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            # Find the finetuned model metrics
            router_eval = eval_data.get("finetuned_router", {})
            cls_metrics = router_eval.get("classification", {})
            routing_metrics = router_eval.get("routing", {})
            
            if cls_metrics:
                metrics = {
                    "accuracy": cls_metrics.get("routing_accuracy", 0.0),
                    "f1_macro": cls_metrics.get("routing_f1_macro", 0.0),
                    "f1_weight": cls_metrics.get("routing_f1", 0.0),
                    "cost_savings": routing_metrics.get("cost_savings_vs_always_strong", 0.0),
                }

    # Load training metrics
    train_report = "reports/experiments/latest_training_result.json"
    if os.path.exists(train_report):
        with open(train_report, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            train_metrics = train_data.get("metrics", {})
            metrics.update({
                "final_train_loss": train_metrics.get("final_train_loss", 0.0),
                "total_training_time_s": train_metrics.get("total_training_time_s", 0.0),
            })

    print("Registering model to MLflow...")
    registry = ModelRegistry()
    
    # 1. Register the model
    reg_result = registry.register_model(
        model_name="cloudops-router",
        model_path=model_path,
        metrics=metrics,
        description="Manually trained and evaluated initial version (v1.0)",
    )
    
    version = reg_result.get("model_version")
    if not version:
        print("ERROR: Failed to register model to MLflow. Check if MLflow server is running on port 5000.")
        return

    # 2. Promote to Production
    print(f"Promoting version {version} to Production...")
    registry.promote_model(
        model_name="cloudops-router",
        version=version,
        target_stage="Production"
    )

    print(f"\n✅ Successfully deployed cloudops-router v{version} to Production!")
    print("You can now restart the API Gateway to use the MLflow model.")

if __name__ == "__main__":
    main()
