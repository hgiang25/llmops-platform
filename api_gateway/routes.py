from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str
    direct_route: str = None  # Optional: force a specific route (e.g., "weak", "strong")

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Option for Direct Routing
    if request.direct_route:
        return {"response": f"Mock response from {request.direct_route} (Direct Route)"}
    
    # 2. Difficulty-Aware Routing
    # Call the Router service (RouteLLM)
    try:
        # Mock calling the router service
        # router_response = requests.post("http://router-service:8001/predict", json={"prompt": request.prompt})
        # difficulty_score = router_response.json().get("difficulty_score", 0.0)
        
        # Mock logic
        difficulty_score = 0.8 if len(request.prompt) > 50 else 0.2
        
        if difficulty_score < 0.5:
            model_path = "Weak Model (Local 7B/8B)"
        else:
            model_path = "Strong Model (Disaggregated Prefill/Decode)"
            
        return {
            "response": f"Mock response processed by {model_path}",
            "difficulty_score": difficulty_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
