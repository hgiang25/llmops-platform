"""
MLOps Pipeline Orchestrator — Điều phối toàn bộ vòng đời MLOps khép kín.

Luồng xử lý chính:
  1. Generate / Collect Data  → Sinh hoặc thu thập dữ liệu CloudOps
  2. Detect Drift             → Kiểm tra sai lệch phân phối dữ liệu
  3. Evaluate Current Model   → Đánh giá hiệu suất model hiện tại
  4. Retrain (if needed)      → Huấn luyện lại model với QLoRA
  5. Evaluate New Model       → Đánh giá model mới
  6. Compare & Register       → So sánh và đăng ký model tốt hơn vào MLflow
  7. Deploy                   → Promote model lên Production stage

Hỗ trợ cả chế độ chạy thủ công (CLI) và trigger qua API Gateway.

📝 LƯU Ý KHI ĐƯA LÊN PRODUCTION:
Hiện tại bước 1 (Generate Data) đang dùng dữ liệu mô phỏng (Synthetic Data/Mocker) để chứng minh khả năng phát hiện Drift. Khi đưa vào thực tế, bạn cần sửa lại bước 1 để truy vấn data thật từ API Gateway Logs (ví dụ: query từ Database hoặc đọc file JSONL sinh ra từ DataCollector).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MLOpsPipeline:
    """
    End-to-end MLOps pipeline orchestrator.

    Coordinates the full closed-loop lifecycle:
    Data → Drift → Evaluate → Train → Register → Deploy.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.pipeline_log = []
        self.status = "idle"  # idle | running | completed | failed

    def _log_step(self, step: str, message: str, data: dict = None):
        """Log a pipeline step."""
        entry = {
            "step": step,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self.pipeline_log.append(entry)
        print(f"[{step}] {message}")

    def run_full_pipeline(
        self,
        force_retrain: bool = False,
        generate_data: bool = True,
    ) -> dict:
        """
        Execute the full MLOps pipeline.

        Args:
            force_retrain: If True, skip drift check and force retraining.
            generate_data: If True, generate synthetic data before running.

        Returns:
            Pipeline execution result dict.
        """
        self.status = "running"
        self.pipeline_log = []
        start_time = time.time()

        print("\n" + "=" * 70)
        print("  MLOPS PIPELINE — CLOSED-LOOP LIFECYCLE EXECUTION")
        print("=" * 70 + "\n")

        try:
            # ----------------------------------------------------------
            # Step 1: Data Generation / Collection
            # ----------------------------------------------------------
            self._log_step("DATA", "Generating synthetic CloudOps data...")
            data_result = self._step_generate_data(generate_data)

            # ----------------------------------------------------------
            # Step 2: Drift Detection
            # ----------------------------------------------------------
            self._log_step("DRIFT", "Running drift detection with Evidently AI...")
            drift_result = self._step_detect_drift()

            drift_detected = drift_result.get("drift_detected", False)
            self._log_step(
                "DRIFT",
                f"Drift detected: {drift_detected} "
                f"(share: {drift_result.get('drift_share', 0)}, "
                f"columns: {drift_result.get('drifted_columns', [])})",
            )

            # ----------------------------------------------------------
            # Step 3: Evaluate Current Model
            # ----------------------------------------------------------
            self._log_step("EVAL", "Evaluating current model...")
            current_eval = self._step_evaluate(model_name="router_current")

            self._log_step(
                "EVAL",
                f"Current model — F1: {current_eval.get('routing_f1', 'N/A')}, "
                f"Cost Savings: {current_eval.get('cost_savings_ratio', 'N/A')}",
            )

            # ----------------------------------------------------------
            # Step 4: Decision — Retrain?
            # ----------------------------------------------------------
            should_retrain = force_retrain or drift_detected

            if not should_retrain:
                self._log_step("DECISION", "No drift detected and no forced retrain. Pipeline complete.")
                self.status = "completed"
                return self._build_result(start_time, retrained=False, drift_result=drift_result,
                                          current_eval=current_eval)

            self._log_step(
                "DECISION",
                f"Retraining triggered! Reason: {'forced' if force_retrain else 'drift detected'}",
            )

            # ----------------------------------------------------------
            # Step 5: Retrain Model (QLoRA Fine-tune)
            # ----------------------------------------------------------
            self._log_step("TRAIN", "Starting QLoRA fine-tuning...")
            train_result = self._step_retrain()

            self._log_step(
                "TRAIN",
                f"Training complete — Final Loss: {train_result.get('final_loss', 'N/A')}, "
                f"Mode: {train_result.get('mode', 'N/A')}",
            )

            # ----------------------------------------------------------
            # Step 6: Evaluate New Model
            # ----------------------------------------------------------
            adapter_path = train_result.get("adapter_path")
            self._log_step("EVAL", f"Evaluating newly trained model from {adapter_path}...")
            # pyrefly: ignore [unexpected-keyword]
            new_eval = self._step_evaluate(model_name="router_retrained", model_path=adapter_path)

            self._log_step(
                "EVAL",
                f"New model — F1: {new_eval.get('routing_f1', 'N/A')}, "
                f"Cost Savings: {new_eval.get('cost_savings_ratio', 'N/A')}",
            )

            # ----------------------------------------------------------
            # Step 7: Compare Models
            # ----------------------------------------------------------
            self._log_step("COMPARE", "Comparing current vs. new model...")
            from mlops.evaluation.evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            comparison = evaluator.compare_models(current_eval, new_eval)

            self._log_step(
                "COMPARE",
                f"Recommendation: {comparison.get('recommendation', 'N/A')} — "
                f"{comparison.get('reason', '')}",
            )

            # ----------------------------------------------------------
            # Step 8: Register & Deploy (if improved)
            # ----------------------------------------------------------
            registration_result = None
            if comparison.get("recommendation") == "DEPLOY":
                self._log_step("REGISTER", "Registering new model to MLflow...")
                registration_result = self._step_register(train_result, new_eval)

                self._log_step(
                    "DEPLOY",
                    f"Model registered: v{registration_result.get('model_version', '?')} — "
                    f"Ready for promotion to Production.",
                )
            else:
                self._log_step("REGISTER", "Keeping current model. No registration needed.")

            self.status = "completed"
            return self._build_result(
                start_time,
                retrained=True,
                drift_result=drift_result,
                current_eval=current_eval,
                new_eval=new_eval,
                train_result=train_result,
                comparison=comparison,
                registration=registration_result,
            )

        except Exception as e:
            self.status = "failed"
            self._log_step("ERROR", f"Pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "pipeline_log": self.pipeline_log,
                "elapsed_time_s": round(time.time() - start_time, 2),
            }

    # ==================================================================
    # Pipeline Steps (each step is independently callable)
    # ==================================================================

    def _step_generate_data(self, generate: bool) -> dict:
        """
        Step 1: Generate or Collect data.
        
        Sử dụng tập dữ liệu thực tế (Real-world Big Data) được cào từ HuggingFace 
        (nằm trong `data/raw/cloudops_real_dataset.jsonl`).
        - ref_data: Nửa đầu của tập dữ liệu (Đại diện cho baseline).
        - current_data: Nửa sau của tập dữ liệu (Đại diện cho log người dùng hôm nay).
        """
        if not generate:
            return {"status": "skipped"}

        import json
        import os
        from pathlib import Path
        
        raw_data_path = "data/raw/cloudops_real_dataset.jsonl"
        
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Không tìm thấy Real Dataset tại {raw_data_path}. Vui lòng chạy hf_data_ingestor.py trước!")
            
        # Read all records from the real dataset
        records = []
        with open(raw_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    
        total_records = len(records)
        if total_records < 2:
            raise ValueError("Dataset quá nhỏ, không đủ để chia đôi!")
            
        # Split into reference (first half) and current (second half)
        mid_point = total_records // 2
        ref_data = records[:mid_point]
        current_data = records[mid_point:]
        
        # Save them to their respective locations
        Path("data/reference").mkdir(parents=True, exist_ok=True)
        with open("data/reference/cloudops_reference.jsonl", "w", encoding="utf-8") as f:
            for record in ref_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        Path("data/current").mkdir(parents=True, exist_ok=True)
        with open("data/current/cloudops_current.jsonl", "w", encoding="utf-8") as f:
            for record in current_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "status": "generated",
            "source": "huggingface_real_data",
            "reference_samples": len(ref_data),
            "current_samples": len(current_data),
        }

    def _step_detect_drift(self) -> dict:
        """Step 2: Run drift detection."""
        from mlops.monitoring.drift_detection import DriftDetector

        detector = DriftDetector()
        return detector.check_drift()

    def _step_evaluate(self, model_name: str = "router_model", model_path: str = None) -> dict:
        """Step 3/6: Evaluate a model."""
        from mlops.evaluation.evaluator import ModelEvaluator
        from mlops.data_pipeline.synthetic_data import generate_dataset
        import os

        evaluator = ModelEvaluator()
        test_data = generate_dataset(n_samples=100, mode="reference", seed=123)
        
        # If no model_path is provided, resolve the current model
        if not model_path:
            from mlops.registry.mlflow_utils import ModelRegistry
            registry = ModelRegistry()
            model_info = registry.load_model(model_name="cloudops-router")
            if "error" not in model_info:
                model_path = model_info.get("source")
            else:
                model_path = "models/cloudops-llm-adapter"

        # Remove URI scheme for local file check
        local_path = model_path.replace("file:///", "").replace("file://", "")
        if not os.path.exists(local_path):
            return {"error": f"Model path {model_path} not found."}

        try:
            predicted_scores = evaluator.predict_scores(local_path, test_data)
            return evaluator.evaluate(test_data, predicted_scores=predicted_scores, model_name=model_name)
        except Exception as e:
            return {"error": str(e)}

    def _step_retrain(self) -> dict:
        """
        Run the QLoRA fine-tuning process on the current dataset.
        Returns the path to the newly trained adapter.
        """
        # pyrefly: ignore [missing-import]
        from mlops.training.finetune import QLoRATrainer, load_train_config

        dataset_path = "data/reference/cloudops_reference.jsonl"
        
        # Initialize and run finetuner
        config = load_train_config()
        finetuner = QLoRATrainer(config=config)
        # Pass self._log_step as callback to stream logs to the UI
        return finetuner.train(dataset_path, callback=self._log_step)

    def _step_register(self, train_result: dict, eval_report: dict) -> dict:
        """Step 8: Register model to MLflow."""
        try:
            from mlops.registry.mlflow_utils import ModelRegistry

            registry = ModelRegistry()

            adapter_path = train_result.get("adapter_path", "models/cloudops-llm-adapter")

            return registry.register_model(
                model_name="cloudops-router",
                model_path=adapter_path,
                metrics={
                    "routing_f1": eval_report.get("routing_f1", 0),
                    "routing_accuracy": eval_report.get("routing_accuracy", 0),
                    "cost_savings_ratio": eval_report.get("cost_savings_ratio", 0),
                    "misrouting_rate": eval_report.get("misrouting_rate", 0),
                    "final_train_loss": train_result.get("final_loss", 0),
                },
                params={
                    "model_name": train_result.get("model_name", "unknown"),
                    "training_mode": train_result.get("mode", "unknown"),
                    "total_epochs": str(train_result.get("total_epochs", 0)),
                    "dataset_samples": str(train_result.get("dataset_samples", 0)),
                },
                description=f"Auto-retrained model. F1={eval_report.get('routing_f1', 0):.4f}",
            )
        except Exception as e:
            self._log_step("REGISTER", f"MLflow registration failed (server may be down): {e}")
            return {
                "status": "registration_failed",
                "error": str(e),
                "note": "Model was trained successfully but could not be registered to MLflow. "
                        "Ensure MLflow server is running (docker-compose up -d).",
            }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _build_result(self, start_time, retrained, drift_result=None,
                      current_eval=None, new_eval=None, train_result=None,
                      comparison=None, registration=None) -> dict:
        """Build the final pipeline result dict."""
        return {
            "status": self.status,
            "retrained": retrained,
            "elapsed_time_s": round(time.time() - start_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_detection": drift_result,
            "current_model_eval": {
                k: current_eval.get(k) for k in
                ["routing_f1", "routing_accuracy", "cost_savings_ratio", "misrouting_rate"]
            } if current_eval else None,
            "new_model_eval": {
                k: new_eval.get(k) for k in
                ["routing_f1", "routing_accuracy", "cost_savings_ratio", "misrouting_rate"]
            } if new_eval else None,
            "training": {
                "final_loss": train_result.get("final_loss"),
                "mode": train_result.get("mode"),
                "adapter_path": train_result.get("adapter_path"),
            } if train_result else None,
            "comparison": comparison,
            "registration": registration,
            "pipeline_log": self.pipeline_log,
        }

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "status": self.status,
            "log_entries": len(self.pipeline_log),
            "last_log": self.pipeline_log[-1] if self.pipeline_log else None,
        }


# Global pipeline instance for API Gateway integration
_pipeline_instance = None


def get_pipeline() -> MLOpsPipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MLOpsPipeline()
    return _pipeline_instance


if __name__ == "__main__":
    pipeline = MLOpsPipeline()
    result = pipeline.run_full_pipeline(force_retrain=True, generate_data=True)

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Status:         {result.get('status', 'unknown')}")
    print(f"  Retrained:      {result.get('retrained', 'N/A')}")
    print(f"  Elapsed Time:   {result.get('elapsed_time_s', 'N/A')}s")

    if result.get("drift_detection"):
        dd = result["drift_detection"]
        print(f"  Drift Detected: {dd.get('drift_detected')}")
        print(f"  Drift Share:    {dd.get('drift_share')}")

    if result.get("current_model_eval"):
        ce = result["current_model_eval"]
        print(f"  Current F1:     {ce.get('routing_f1')}")

    if result.get("new_model_eval"):
        ne = result["new_model_eval"]
        print(f"  New F1:         {ne.get('routing_f1')}")

    if result.get("comparison"):
        comp = result["comparison"]
        print(f"  Recommendation: {comp.get('recommendation')}")

    if result.get("registration"):
        reg = result["registration"]
        print(f"  Model Version:  {reg.get('model_version', 'N/A')}")

    print("=" * 70)
