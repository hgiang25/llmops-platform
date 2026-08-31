from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import time
import traceback

from mlops.data_pipeline.collector import DataCollector
from mlops.pipeline import get_pipeline

router = APIRouter()

# Initialize shared DataCollector (singleton-like)
data_collector = DataCollector(log_dir="data/raw")


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
    Now integrated with DataCollector to log every request.
    """
    start_time = time.time()

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
        # Mock difficulty scoring (replace with actual Router model later)
        difficulty_score = 0.8 if len(request.prompt) > 50 else 0.2

        if difficulty_score < 0.4:
            model_path = "Weak Model (Local 7B/8B)"
            route = "weak"
            model_used = "cloudops-llm-7b-finetuned"
        elif difficulty_score < 0.7:
            model_path = "Strong Model (Disaggregated Prefill/Decode)"
            route = "strong_disaggregated"
            model_used = "cloudops-llm-70b-disaggregated"
        else:
            model_path = "Strong Model (External API)"
            route = "strong_external"
            model_used = "gpt-4o"

        response_text = f"Mock response processed by {model_path}"
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
