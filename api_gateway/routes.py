# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, BackgroundTasks
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional
import time
import traceback
import asyncio
import requests
import os

from mlops.data_pipeline.collector import DataCollector
from mlops.pipeline import get_pipeline

router = APIRouter()

# Initialize shared DataCollector (singleton-like)
data_collector = DataCollector(log_dir="data/raw")

# =====================================================================
# Router Model Predictor (Singleton)
# =====================================================================

class RouterPredictor:
    """
    Difficulty-aware router backed by the DeBERTa-v3 + Ordinal head model
    trained on UltraFeedback-derived routing labels (data/splits/train.jsonl).

    Resolution order:
      1. MLflow registry model "cloudops-router" (if it contains model_config.json + model.pt)
      2. Local trained model at models/deberta-router
      3. Heuristic fallback (length-based) — flagged as mode="heuristic_fallback"
    """
    LOCAL_MODEL_DIR = os.environ.get("ROUTER_MODEL_DIR", "models/deberta-router")

    _model = None
    _tokenizer = None
    _device = None
    _max_len = 256
    _is_loaded = False
    mode = "not_loaded"
    model_source = None

    @staticmethod
    def _is_valid_model_dir(path: str) -> bool:
        import json
        cfg = os.path.join(path, "model_config.json")
        if not (os.path.exists(cfg) and os.path.exists(os.path.join(path, "model.pt"))):
            return False
        with open(cfg, "r", encoding="utf-8") as f:
            c = json.load(f)
        return c.get("architecture") == "ordinal_classification" and not c.get("mock_training", False)

    @classmethod
    def _resolve_model_dir(cls) -> Optional[str]:
        try:
            from mlops.registry.mlflow_utils import ModelRegistry
            info = ModelRegistry().load_model(model_name="cloudops-router")
            if "error" not in info:
                import mlflow
                local = mlflow.artifacts.download_artifacts(artifact_uri=info.get("source"))
                if cls._is_valid_model_dir(local):
                    cls.model_source = f"mlflow:cloudops-router/v{info.get('version')}"
                    return local
                print(f"[API] MLflow model v{info.get('version')} is not a trained DeBERTa ordinal router. Ignoring.")
        except Exception as e:
            print(f"[API] MLflow unavailable ({e}).")

        if cls._is_valid_model_dir(cls.LOCAL_MODEL_DIR):
            cls.model_source = f"local:{cls.LOCAL_MODEL_DIR}"
            return cls.LOCAL_MODEL_DIR
        return None

    @classmethod
    def load(cls):
        if cls._is_loaded:
            return
        import json
        model_dir = cls._resolve_model_dir()
        if model_dir is None:
            print("[API] No trained router found. Run: python scripts/train_router.py — using heuristic fallback.")
            cls.mode = "heuristic_fallback"
            cls._is_loaded = True
            return

        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        from transformers import AutoTokenizer
        from mlops.training.ordinal_model import DeBERTaOrdinalClassifier

        with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model", "microsoft/deberta-v3-base")
        head_cfg = cfg.get("ordinal_head", {})

        print(f"[API] Loading DeBERTa ordinal router from {model_dir} ...")
        tok_src = model_dir if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")) else base_model
        cls._tokenizer = AutoTokenizer.from_pretrained(tok_src)
        model = DeBERTaOrdinalClassifier(
            model_name=base_model,
            hidden_dim=head_cfg.get("hidden_dim", 256),
            n_classes=cfg.get("num_labels", 3),
            dropout=head_cfg.get("dropout", 0.1),
        )
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._model = model.to(cls._device).eval()
        cls.mode = "deberta_ordinal"
        cls._is_loaded = True
        print(f"[API] Router ready on {cls._device} (source={cls.model_source}).")

    @classmethod
    def predict(cls, prompt: str) -> dict:
        """Return {'label': 0|1|2, 'probs': [p0,p1,p2], 'difficulty_score': float}."""
        cls.load()
        if cls.mode == "heuristic_fallback":
            label = 0 if len(prompt) < 50 else 2
            probs = [1.0 if i == label else 0.0 for i in range(3)]
            return {"label": label, "probs": probs, "difficulty_score": label / 2.0}

        # pyrefly: ignore [missing-import]
        import torch
        inputs = cls._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=cls._max_len
        ).to(cls._device)
        with torch.no_grad():
            out = cls._model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probs = out["probs"][0].float().cpu().tolist()
        label = int(out["predictions"][0].item())
        import math
        if any(math.isnan(p) for p in probs):
            print("[API] Warning: Router returned NaN probabilities. Fallback to Weak mode.")
            probs = [1.0, 0.0, 0.0]
            label = 0
            
        # Expected ordinal difficulty in [0, 1]
        difficulty = (probs[1] * 1 + probs[2] * 2) / 2.0
        return {"label": label, "probs": [round(p, 4) for p in probs], "difficulty_score": round(difficulty, 4)}




# =====================================================================
# Request / Response Models
# =====================================================================

class ChatRequest(BaseModel):
    prompt: str
    direct_route: str = None  # Optional: force a specific route (e.g., "weak", "strong")


class FeedbackRequest(BaseModel):
    thumbs_up: bool = False
    thumbs_down: bool = False
    regenerate: bool = False


class MLOpsTriggerRequest(BaseModel):
    force_retrain: bool = False
    generate_data: bool = True


# =====================================================================
# Chat Endpoint (with Data Logging integration)
# =====================================================================

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with Difficulty-Aware Routing.
    Now integrated with actual Model Inference.
    """
    start_time = time.time()

    async def call_openai_compatible_api(url: str, model_name: str, prompt: str, api_key: str = None):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        def _post():
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
            
        try:
            result = await asyncio.to_thread(_post)
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            # Fallback for local development when vLLM/SGLang is not running
            return f"(Mô phỏng) Mô hình {model_name} đang phản hồi... Vui lòng bật server vLLM/SGLang tại {url} để nhận phản hồi thật."
        except Exception as e:
            return f"[Error calling model {model_name} at {url}]: {str(e)}"

    # 1. Option for Direct Routing
    if request.direct_route:
        response_text = f"Mock response from {request.direct_route} (Direct Route)"
        elapsed = (time.time() - start_time) * 1000

        # Log the request
        record = data_collector.log_request(
            prompt=request.prompt,
            route=request.direct_route,
            difficulty_score=0.0,
            model_used=f"{request.direct_route}_direct",
            response_time_ms=round(elapsed, 2),
            response_text=response_text,
        )
        return {"response": response_text, "log_id": record["log_id"]}

    # 2. Difficulty-Aware Routing
    try:
        # Use real Router model (DeBERTa ordinal, trained on UltraFeedback)
        router_out = await asyncio.to_thread(RouterPredictor.predict, request.prompt)
        predicted_class = router_out["label"]
        difficulty_score = router_out["difficulty_score"]

        if predicted_class == 0:
            model_path = "Weak Model (vLLM Qwen 0.5B)"
            route = "weak"
            model_used = "Qwen/Qwen2.5-0.5B-Instruct"
            response_text = await call_openai_compatible_api("http://localhost:8001/v1/chat/completions", model_used, request.prompt)
        elif predicted_class == 1:
            model_path = "Strong Model (SGLang Qwen 0.5B)"
            route = "strong_disaggregated"
            model_used = "Qwen/Qwen2.5-0.5B-Instruct"
            response_text = await call_openai_compatible_api("http://localhost:8002/v1/chat/completions", model_used, request.prompt)
        else:
            model_path = "Strong Model (External API GPT-4o)"
            route = "strong_external"
            model_used = "gpt-4o"
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                response_text = await call_openai_compatible_api(
                    "https://api.openai.com/v1/chat/completions", 
                    model_used, 
                    request.prompt, 
                    api_key=openai_api_key
                )
            else:
                response_text = "Mock response: Vui lòng cấu hình biến môi trường OPENAI_API_KEY để dùng gpt-4o thật cho các câu hỏi cực khó."

        elapsed = (time.time() - start_time) * 1000

        # Log the request to DataCollector
        record = data_collector.log_request(
            prompt=request.prompt,
            route=route,
            difficulty_score=difficulty_score,
            model_used=model_used,
            response_time_ms=round(elapsed, 2),
            response_text=response_text,
        )

        return {
            "response": response_text,
            "difficulty_score": difficulty_score,
            "route": route,
            "model_used": model_used,
            "router_mode": RouterPredictor.mode,
            "router_source": RouterPredictor.model_source,
            "router_probs": router_out["probs"],
            "log_id": record["log_id"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/router/info")
async def router_info():
    """Show which router model is currently serving (trained DeBERTa vs heuristic fallback)."""
    await asyncio.to_thread(RouterPredictor.load)
    return {"mode": RouterPredictor.mode, "source": RouterPredictor.model_source}

@router.post("/chat/{log_id}/feedback")
async def submit_feedback(log_id: str, request: FeedbackRequest):
    """Submit user feedback for a specific chat response."""
    updated = data_collector.update_feedback(
        log_id=log_id,
        thumbs_up=request.thumbs_up,
        thumbs_down=request.thumbs_down,
        regenerate=request.regenerate
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Log entry not found")
    return {"status": "success", "message": "Feedback recorded"}


# =====================================================================
# MLOps Endpoints
# =====================================================================

@router.get("/mlops/status")
async def mlops_status():
    """Get current MLOps pipeline status and data collection stats."""
    pipeline = get_pipeline()
    stats = data_collector.get_stats()
    pipeline_status = pipeline.get_status()

    return {
        "pipeline": pipeline_status,
        "data_collection": stats,
    }


@router.post("/mlops/check-drift")
async def mlops_check_drift():
    """
    Trigger drift detection between reference and current datasets.
    Requires synthetic data to be generated first.
    """
    try:
        from mlops.monitoring.drift_detection import DriftDetector

        detector = DriftDetector()
        result = detector.check_drift()
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Data files not found. Generate data first via /mlops/run-pipeline. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@router.post("/mlops/retrain")
async def mlops_retrain(
    request: MLOpsTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the full MLOps pipeline (Data → Drift → Eval → Train → Register).
    Runs in background to avoid HTTP timeout.
    """
    pipeline = get_pipeline()

    if pipeline.status == "running":
        return {
            "status": "already_running",
            "message": "Pipeline is already running. Check /mlops/status for progress.",
        }

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline_background,
        pipeline,
        request.force_retrain,
        request.generate_data,
    )

    return {
        "status": "started",
        "message": "MLOps pipeline started in background. Check /mlops/status for progress.",
        "force_retrain": request.force_retrain,
    }


@router.post("/mlops/run-pipeline")
async def mlops_run_pipeline_sync(request: MLOpsTriggerRequest):
    """
    Run the full MLOps pipeline synchronously (for demo/testing).
    Warning: This may take a while depending on training configuration.
    """
    pipeline = get_pipeline()

    if pipeline.status == "running":
        return {
            "status": "already_running",
            "message": "Pipeline is already running.",
        }

    try:
        result = pipeline.run_full_pipeline(
            force_retrain=request.force_retrain,
            generate_data=request.generate_data,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}\n{traceback.format_exc()}")


@router.get("/mlops/data-stats")
async def mlops_data_stats():
    """Get statistics about collected data."""
    return data_collector.get_stats()


@router.get("/mlops/logs")
async def mlops_logs(last_n: int = 20):
    """Get recent data collection logs."""
    logs = data_collector.load_logs(last_n=last_n)
    return {"logs": logs, "count": len(logs)}


@router.get("/mlops/pipeline-log")
async def mlops_pipeline_log():
    """Get the pipeline execution log."""
    pipeline = get_pipeline()
    return {
        "status": pipeline.status,
        "log": pipeline.pipeline_log,
    }


# =====================================================================
# Background task helper
# =====================================================================

def _run_pipeline_background(pipeline, force_retrain: bool, generate_data: bool):
    """Execute the MLOps pipeline as a background task."""
    try:
        pipeline.run_full_pipeline(
            force_retrain=force_retrain,
            generate_data=generate_data,
        )
    except Exception as e:
        pipeline.status = "failed"
        pipeline._log_step("ERROR", f"Background pipeline failed: {str(e)}")
        
