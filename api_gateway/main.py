from fastapi import FastAPI
from api_gateway.routes import router

app = FastAPI(
    title="LLMOps Platform API Gateway",
    description="API Gateway for Difficulty-Aware Routing and Inference",
    version="1.0.0"
)

from prometheus_client import make_asgi_app

app.include_router(router)

# Mount Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
