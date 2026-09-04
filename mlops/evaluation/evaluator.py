"""
Model Evaluator — Đánh giá hiệu suất mô hình Router (offline evaluation).

Đánh giá khả năng phân loại độ khó của Router model bằng các metrics:
  - Accuracy, Precision, Recall, F1 Score
  - Cost Savings Ratio (% requests gửi đến weak model mà vẫn đúng)
  - Confusion Matrix

So sánh hiệu suất giữa current model và newly trained model để quyết định
có nên triển khai model mới hay không.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ModelEvaluator:
    """
    Evaluates Router model performance on a held-out test set.

    The Router model predicts difficulty scores (0–1) which are then
    thresholded into routing decisions:
      - score < T_low  → weak model
      - score >= T_high → strong model (external API)
      - T_low <= score < T_high → strong model (disaggregated)
    """

    def __init__(
        self,
        t_low: float = 0.4,
        t_high: float = 0.7,
        report_output_dir: str = "data/reports",
    ):
        self.t_low = t_low
        self.t_high = t_high
        self.report_output_dir = Path(report_output_dir)
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

    def _score_to_route(self, score: float) -> str:
        """Convert a difficulty score into a routing decision."""
        if score < self.t_low:
            return "weak"
        elif score < self.t_high:
            return "strong_disaggregated"
        else:
            return "strong_external"

    def _score_to_binary(self, score: float) -> int:
        """Convert difficulty score to binary: 0=weak, 1=strong."""
        return 0 if score < self.t_low else 1

    def evaluate(
        self,
        test_data: list[dict],
        predicted_scores: Optional[list[float]] = None,
        model_name: str = "router_model",
    ) -> dict:
        """
        Evaluate router model predictions against ground truth.

        Args:
            test_data: List of records with 'difficulty_score' (ground truth)
                       and 'domain' fields.
            predicted_scores: Model's predicted difficulty scores.
                              If None, uses a simulated predictor.
            model_name: Name of the model being evaluated.

        Returns:
            Evaluation report dict with all metrics.
        """
        if not test_data:
            return {"error": "No test data provided."}

        # Ground truth routing decisions
        true_scores = [r["difficulty_score"] for r in test_data]
        true_routes = [self._score_to_route(s) for s in true_scores]
        true_binary = [self._score_to_binary(s) for s in true_scores]

        # Predicted routing decisions
        if predicted_scores is None:
            return {"error": "predicted_scores must be provided (run predict_scores first)."}

        pred_routes = [self._score_to_route(s) for s in predicted_scores]
        pred_binary = [self._score_to_binary(s) for s in predicted_scores]

        # --- Compute metrics ---

        # Multi-class metrics (weak / strong_disaggregated / strong_external)
        all_labels = ["weak", "strong_disaggregated", "strong_external"]
        accuracy = accuracy_score(true_routes, pred_routes)
        precision = precision_score(
            true_routes, pred_routes, labels=all_labels, average="weighted", zero_division=0
        )
        recall = recall_score(
            true_routes, pred_routes, labels=all_labels, average="weighted", zero_division=0
        )
        f1 = f1_score(
            true_routes, pred_routes, labels=all_labels, average="weighted", zero_division=0
        )

        # Binary metrics (weak vs strong)
        binary_accuracy = accuracy_score(true_binary, pred_binary)
        binary_f1 = f1_score(true_binary, pred_binary, zero_division=0)

        # Cost Savings Ratio:
        # % of requests correctly routed to weak model
        weak_correct = sum(
            1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0
        )
        total_truly_weak = sum(1 for t in true_binary if t == 0)
        cost_savings_ratio = (
            weak_correct / total_truly_weak if total_truly_weak > 0 else 0.0
        )

        # Misrouting cost:
        # Hard queries sent to weak model (potential quality loss)
        misrouted_hard = sum(
            1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0
        )
        misrouting_rate = misrouted_hard / len(true_binary) if true_binary else 0.0

        # Confusion matrix
        cm = confusion_matrix(true_routes, pred_routes, labels=all_labels)

        # Classification report
        cls_report = classification_report(
            true_routes, pred_routes, labels=all_labels, output_dict=True, zero_division=0
        )

        # Score distribution stats
        score_errors = [
            abs(t - p) for t, p in zip(true_scores, predicted_scores)
        ]
        mae = np.mean(score_errors)

        report = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(test_data),
            "thresholds": {"t_low": self.t_low, "t_high": self.t_high},
            # Multi-class routing metrics
            "routing_accuracy": round(accuracy, 4),
            "routing_precision": round(precision, 4),
            "routing_recall": round(recall, 4),
            "routing_f1": round(f1, 4),
            # Binary (weak vs strong) metrics
            "binary_accuracy": round(binary_accuracy, 4),
            "binary_f1": round(binary_f1, 4),
            # Cost metrics
            "cost_savings_ratio": round(cost_savings_ratio, 4),
            "misrouting_rate": round(misrouting_rate, 4),
            "misrouted_hard_queries": misrouted_hard,
            # Score prediction quality
            "score_mae": round(float(mae), 4),
            # Confusion matrix
            "confusion_matrix": {
                "labels": all_labels,
                "matrix": cm.tolist(),
            },
            # Per-class report
            "classification_report": cls_report,
        }

        # Save report
        self._save_report(report)

        return report

    def compare_models(
        self,
        current_report: dict,
        new_report: dict,
        improvement_threshold: float = 0.02,
    ) -> dict:
        """
        Compare two evaluation reports to decide if the new model is better.

        Args:
            current_report: Evaluation report of the currently deployed model.
            new_report: Evaluation report of the newly trained model.
            improvement_threshold: Minimum F1 improvement to recommend deployment.

        Returns:
            Comparison result dict with recommendation.
        """
        current_f1 = current_report.get("routing_f1", 0)
        new_f1 = new_report.get("routing_f1", 0)
        f1_delta = new_f1 - current_f1

        current_cost = current_report.get("cost_savings_ratio", 0)
        new_cost = new_report.get("cost_savings_ratio", 0)
        cost_delta = new_cost - current_cost

        current_misroute = current_report.get("misrouting_rate", 1)
        new_misroute = new_report.get("misrouting_rate", 1)

        # Decision logic
        should_deploy = (
            f1_delta >= improvement_threshold
            and new_misroute <= current_misroute + 0.01
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_model": current_report.get("model_name", "current"),
            "new_model": new_report.get("model_name", "new"),
            "current_f1": current_f1,
            "new_f1": new_f1,
            "f1_improvement": round(f1_delta, 4),
            "current_cost_savings": current_cost,
            "new_cost_savings": new_cost,
            "cost_savings_improvement": round(cost_delta, 4),
            "current_misrouting_rate": current_misroute,
            "new_misrouting_rate": new_misroute,
            "recommendation": "DEPLOY" if should_deploy else "KEEP_CURRENT",
            "reason": (
                f"New model F1 improved by {f1_delta:.4f} (>= {improvement_threshold})"
                if should_deploy
                else f"F1 improvement {f1_delta:.4f} below threshold {improvement_threshold} "
                     f"or misrouting rate increased"
            ),
        }

    def predict_scores(self, model_path: str, test_data: list[dict]) -> list[float]:
        """
        Run actual inference using the fine-tuned model to predict difficulty scores.
        """
        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[EVAL] Loading model for inference from {model_path}...")
        
        # Handle MLflow file:/// artifact URIs
        if model_path.startswith("file:///"):
            model_path = model_path[8:]
        elif model_path.startswith("file://"):
            model_path = model_path[7:]
            
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()

        predicted_scores = []
        print(f"[EVAL] Running inference on {len(test_data)} samples...")
        
        for r in test_data:
            prompt = r.get("prompt", "")
            input_text = (
                f"### Instruction:\n{prompt}\n\n"
                f"### Score:\n"
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    do_sample=False,
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract score from generated text
            try:
                # The score should be the first float in the generated text
                words = generated_text.strip().split()
                if not words:
                    score = 0.5
                else:
                    score = float(words[0])
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 0.5  # default on parsing failure
                
            predicted_scores.append(score)
            
        return predicted_scores


    def _save_report(self, report: dict):
        """Save evaluation report as JSON."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = report.get("model_name", "model")
        filepath = self.report_output_dir / f"eval_{model_name}_{timestamp_str}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Evaluation report saved to {filepath}")


if __name__ == "__main__":
    from mlops.data_pipeline.synthetic_data import generate_dataset

    # Generate test data
    test_data = generate_dataset(n_samples=100, mode="reference", seed=123)

    evaluator = ModelEvaluator()
    report = evaluator.evaluate(test_data, model_name="router_v1_baseline")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Model:              {report['model_name']}")
    print(f"  Samples:            {report['n_samples']}")
    print(f"  Routing Accuracy:   {report['routing_accuracy']}")
    print(f"  Routing F1:         {report['routing_f1']}")
    print(f"  Binary Accuracy:    {report['binary_accuracy']}")
    print(f"  Cost Savings Ratio: {report['cost_savings_ratio']}")
    print(f"  Misrouting Rate:    {report['misrouting_rate']}")
    print(f"  Score MAE:          {report['score_mae']}")
    print("=" * 60)
