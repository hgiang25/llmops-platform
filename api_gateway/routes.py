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
    _model = None
    _tokenizer = None
    _is_loaded = False

    @classmethod
    def predict(cls, prompt: str) -> float:
        if not cls._is_loaded:
            from mlops.registry.mlflow_utils import ModelRegistry
            registry = ModelRegistry()
            model_info = registry.load_model(model_name="cloudops-router")
            
            if "error" in model_info:
                print(f"[API] MLflow load failed ({model_info['error']}). Falling back to local adapter.")
                model_path = "models/cloudops-llm-adapter"
                if not os.path.exists(model_path):
                    print("[API] Local adapter not found. Using fallback mock.")
                    return 0.8 if len(prompt) > 50 else 0.2
            else:
                model_path = model_info.get("source")
                if model_path.startswith("file:///"):
                    model_path = model_path[8:]
                elif model_path.startswith("file://"):
                    model_path = model_path[7:]
                
            # pyrefly: ignore [missing-import]
            import torch
            # pyrefly: ignore [missing-import]
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[API] Loading router model from {model_path}...")
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if cls._tokenizer.pad_token is None:
                cls._tokenizer.pad_token = cls._tokenizer.eos_token
                
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            cls._model.eval()
            cls._is_loaded = True

        input_text = f"### Instruction:\n{prompt}\n\n### Score:\n"
        inputs = cls._tokenizer(input_text, return_tensors="pt").to(cls._model.device)
        
        # pyrefly: ignore [missing-import]
        import torch
        with torch.no_grad():
            outputs = cls._model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=cls._tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False,
            )
        generated_text = cls._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            words = generated_text.strip().split()
            score = float(words[0]) if words else 0.5
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5



# =====================================================================
# Request / Response Models
# =====================================================================

class ChatRequest(BaseModel):
    prompt: str
    direct_route: str = None  # Optional: force a specific route (e.g., "weak", "strong")


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
        data_collector.log_request(
            prompt=request.prompt,
            route=request.direct_route,
            difficulty_score=0.0,
            model_used=f"{request.direct_route}_direct",
            response_time_ms=round(elapsed, 2),
            response_text=response_text,
        )
        return {"response": response_text}

    # 2. Difficulty-Aware Routing
    try:
        # Use real Router model (or fallback if Day 0)
        difficulty_score = await asyncio.to_thread(RouterPredictor.predict, request.prompt)

        if difficulty_score < 0.4:
            model_path = "Weak Model (vLLM Qwen 0.5B)"
            route = "weak"
            model_used = "Qwen/Qwen2.5-0.5B-Instruct"
            response_text = await call_openai_compatible_api("http://localhost:8001/v1/chat/completions", model_used, request.prompt)
        elif difficulty_score < 0.7:
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
        data_collector.log_request(
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
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
