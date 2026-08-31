# LLMOps Platform (Architecture v3)

This repository contains the full end-to-end LLMOps Platform focusing on **Cost- and Workload-Aware Resource Orchestration for Specialized LLM Inference**.

## Architecture Components

1. **Frontend**: Streamlit Web UI for management and interaction.
2. **API Gateway**: FastAPI service routing user requests.
3. **Routing Layer**: Difficulty-Aware Routing based on RouteLLM (LightGBM/BERT).
4. **Inference Core**: 
   - **Weak Model**: Local 7B/8B inference.
   - **Strong Model**: Disaggregated Inference (Prefill/Decode) with LMCache.
5. **Orchestration**: Ray Serve & KubeRay for fractional GPUs and dynamic scaling.
6. **Observability**: Prometheus & Grafana for system metrics.
7. **MLOps Lifecycle**: Evidently AI for drift detection, MLflow for model registry, QLoRA for training.

## Quick Start

### 1. Environment Setup

**For Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
docker-compose up -d  # Start Redis, Prometheus, Grafana, MLflow
```

**For Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
docker-compose up -d  # Start Redis, Prometheus, Grafana, MLflow
```

### 2. Run the Application

You need to open **2 separate terminals** and activate the virtual environment in both before running the services.

**Terminal 1 (Run API Gateway):**
```powershell
.\venv\Scripts\activate
uvicorn api_gateway.main:app --reload --port 8000
```

**Terminal 2 (Run Streamlit Frontend):**
```powershell
.\venv\Scripts\activate
streamlit run frontend/app.py
```
