"""
MLflow Utilities — Quản lý Model Registry và Experiment Tracking.

Cung cấp interface thống nhất để:
  - Đăng ký model mới vào MLflow Model Registry
  - Log training metrics, parameters, và artifacts
  - Load model theo version hoặc stage (Staging/Production)
  - Promote model giữa các stages
  - Truy vấn danh sách model versions

Thiết kế linh hoạt:
  - Local: Kết nối MLflow server chạy trên Docker (localhost:5000)
  - Cloud: Chỉ cần thay đổi tracking_uri sang cloud endpoint
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
from mlflow.tracking import MlflowClient


class ModelRegistry:
    """
    Unified interface for MLflow Model Registry operations.

    Handles experiment tracking, model registration, versioning,
    and stage management (None → Staging → Production → Archived).
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        default_experiment: str = "cloudops-llm-finetuning",
    ):
        self.tracking_uri = tracking_uri
        self.default_experiment = default_experiment

        # Set S3/MinIO credentials for artifact logging
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minio_admin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio_password123")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Ensure default experiment exists
        experiment = mlflow.get_experiment_by_name(self.default_experiment)
        if experiment is None:
            mlflow.create_experiment(self.default_experiment)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Optional[dict] = None,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
        experiment_name: Optional[str] = None,
        description: str = "",
    ) -> dict:
        """
        Register a model (or adapter) to MLflow Model Registry.

        This creates an MLflow Run, logs all artifacts/metrics/params,
        and registers the model as a new version.

        Args:
            model_name: Registry model name (e.g., "cloudops-router", "cloudops-llm-adapter").
            model_path: Path to model directory (adapter weights, config, etc.).
            metrics: Training/evaluation metrics to log.
            params: Training parameters to log.
            tags: Additional tags for the run.
            experiment_name: MLflow experiment name (defaults to self.default_experiment).
            description: Human-readable description of this model version.

        Returns:
            Dict with run_id, model_version, and model_uri.
        """
        experiment = experiment_name or self.default_experiment
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            if params:
                for key, value in params.items():
                    # MLflow params must be strings with max 500 chars
                    mlflow.log_param(key, str(value)[:500])

            # Log metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)

            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, str(value))

            # Always tag with timestamp and model type
            mlflow.set_tag("registered_at", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("model_type", model_name)

            # Log model artifacts
            model_dir = Path(model_path)
            if model_dir.exists():
                if model_dir.is_dir():
                    mlflow.log_artifacts(str(model_dir), artifact_path="model")
                else:
                    mlflow.log_artifact(str(model_dir), artifact_path="model")

            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                result = mlflow.register_model(model_uri, model_name)
                model_version = result.version
            except Exception as e:
                print(f"[WARNING] Model registration failed: {e}")
                model_version = None

            # Add description to version
            if model_version and description:
                try:
                    self.client.update_model_version(
                        name=model_name,
                        version=model_version,
                        description=description,
                    )
                except Exception:
                    pass

            registration_result = {
                "status": "registered",
                "model_name": model_name,
                "model_version": model_version,
                "run_id": run.info.run_id,
                "model_uri": model_uri,
                "experiment": experiment,
                "artifact_path": str(model_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            print(f"\n[OK] Model registered: {model_name} v{model_version}")
            print(f"  Run ID:     {run.info.run_id}")
            print(f"  Model URI:  {model_uri}")

            return registration_result

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> dict:
        """
        Load model info from the registry.

        Args:
            model_name: Registered model name.
            version: Specific version number. Takes precedence over stage.
            stage: Model stage ("Staging", "Production").

        Returns:
            Dict with model info, version, and artifact URI.
        """
        try:
            if version:
                mv = self.client.get_model_version(model_name, version)
            elif stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if not versions:
                    return {"error": f"No model found in stage '{stage}' for '{model_name}'."}
                mv = versions[0]
            else:
                mv = self._get_latest_version_obj(model_name)
                if mv is None:
                    return {"error": f"No versions found for model '{model_name}'."}

            return {
                "model_name": mv.name,
                "version": mv.version,
                "stage": mv.current_stage,
                "status": mv.status,
                "source": mv.source,
                "run_id": mv.run_id,
                "description": mv.description or "",
                "creation_timestamp": mv.creation_timestamp,
            }
        except Exception as e:
            return {"error": str(e)}

    def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: str = "Production",
        archive_existing: bool = True,
    ) -> dict:
        """
        Promote a model version to a target stage.

        Args:
            model_name: Registered model name.
            version: Version to promote.
            target_stage: Target stage ("Staging" or "Production").
            archive_existing: If True, archive the current model in that stage.

        Returns:
            Dict with promotion details.
        """
        try:
            # Archive existing models in the target stage
            if archive_existing:
                existing = self.client.get_latest_versions(model_name, stages=[target_stage])
                for mv in existing:
                    if mv.version != version:
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Archived",
                        )
                        print(f"  Archived: {model_name} v{mv.version}")

            # Promote the target version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=target_stage,
            )

            print(f"[OK] Promoted {model_name} v{version} -> {target_stage}")
            return {
                "status": "promoted",
                "model_name": model_name,
                "version": version,
                "stage": target_stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version number for a registered model."""
        mv = self._get_latest_version_obj(model_name)
        return mv.version if mv else None

    def list_versions(self, model_name: str) -> list[dict]:
        """List all versions of a registered model."""
        try:
            # Use search_model_versions for compatibility
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "status": mv.status,
                    "description": mv.description or "",
                    "creation_timestamp": mv.creation_timestamp,
                    "run_id": mv.run_id,
                }
                for mv in versions
            ]
        except Exception as e:
            print(f"[WARNING] Could not list versions: {e}")
            return []

    def list_experiments(self) -> list[dict]:
        """List all MLflow experiments."""
        experiments = self.client.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
            }
            for exp in experiments
        ]

    def log_evaluation(
        self,
        model_name: str,
        eval_report: dict,
        experiment_name: Optional[str] = None,
    ) -> str:
        """
        Log an evaluation report as a dedicated MLflow run.

        Returns the run_id.
        """
        experiment = experiment_name or self.default_experiment
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=f"eval_{model_name}_{datetime.now().strftime('%H%M%S')}") as run:
            mlflow.set_tag("run_type", "evaluation")
            mlflow.set_tag("model_name", model_name)

            # Log scalar metrics
            for key in ["routing_accuracy", "routing_f1", "binary_accuracy",
                         "cost_savings_ratio", "misrouting_rate", "score_mae"]:
                if key in eval_report:
                    mlflow.log_metric(key, eval_report[key])

            # Log full report as artifact
            report_path = Path("data/reports") / f"eval_{model_name}_temp.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(eval_report, f, indent=2)
            mlflow.log_artifact(str(report_path))

            return run.info.run_id

    def _get_latest_version_obj(self, model_name: str):
        """Internal: get the ModelVersion object for the latest version."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return None
            # Sort by version number descending
            versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
            return versions_sorted[0]
        except Exception:
            return None


def register_model(model_name: str, model_path: str, **kwargs):
    """Convenience function for CLI usage."""
    registry = ModelRegistry()
    return registry.register_model(model_name, model_path, **kwargs)


if __name__ == "__main__":
    # Demo: List experiments and register a mock model
    registry = ModelRegistry()

    print("MLflow Experiments:")
    for exp in registry.list_experiments():
        print(f"  [{exp['experiment_id']}] {exp['name']}")

    # Register a mock adapter
    mock_path = Path("models/cloudops-llm-adapter")
    if mock_path.exists():
        result = registry.register_model(
            model_name="cloudops-router",
            model_path=str(mock_path),
            metrics={"routing_f1": 0.89, "cost_savings_ratio": 0.65},
            params={"lora_r": "16", "lora_alpha": "32", "epochs": "3"},
            description="Baseline router model with QLoRA adapter",
        )
        print(f"\nRegistration result: {json.dumps(result, indent=2)}")
    else:
        print(f"\nNo model found at {mock_path}. Run training first.")
