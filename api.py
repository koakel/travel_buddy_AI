from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from main import TravelCompanionAI

# Pydantic models for request/response bodies
class QueryRequest(BaseModel):
    user_input: str
    user_profile: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str

# Initialize FastAPI app
app = FastAPI(title="Travel Companion AI API",
              description="API for the AI travel companion for elderly users",
              version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key from environment or use default
PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
API_KEY = os.getenv(f"{PROVIDER.upper()}_API_KEY", "sk-45b1f23aa71d423d90645988ef3d1d22")
API_BASE = os.getenv(f"{PROVIDER.upper()}_API_BASE")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# Initialize the AI companion (non-interactive mode)
try:
    ai_companion = TravelCompanionAI(
        provider=PROVIDER,
        api_key=API_KEY,
        api_base=API_BASE,
        model_name=MODEL_NAME,
        skip_interactive=True
    )
    print(f"AI Companion initialized successfully using {PROVIDER}.")
except Exception as e:
    print(f"Error initializing AI Companion: {e}")
    ai_companion = None # Set to None if initialization fails

# Root path redirects to docs
@app.get("/", include_in_schema=False)
async def root():
    """Root path redirects to API documentation."""
    return RedirectResponse(url="/docs")

# API Endpoint
@app.post("/query", response_model=QueryResponse)
def process_user_query(request: QueryRequest):
    """
    Process a user's query and return the AI's response.
    """
    if ai_companion is None:
        raise HTTPException(status_code=500, detail="AI Companion failed to initialize.")

    print(f"Received query: {request.user_input}")
    
    # Update user profile if provided in the request
    current_profile = ai_companion.user_profile.copy()
    if request.user_profile:
        current_profile.update(request.user_profile)
        print(f"Using updated user profile: {current_profile}")
    else:
        print(f"Using default user profile: {current_profile}")
        
    try:
        # Override the companion's profile for this request
        original_profile = ai_companion.user_profile
        ai_companion.user_profile = current_profile
        
        # Process the query using the AI companion
        response_text = ai_companion.process_query(request.user_input)
        
        # Restore original profile
        ai_companion.user_profile = original_profile
        
        print(f"Generated response: {response_text[:100]}...")
        return QueryResponse(response=response_text)
    except Exception as e:
        # Restore original profile even if error occurs
        ai_companion.user_profile = original_profile
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API is running."""
    if ai_companion is None:
        return {"status": "error", "detail": "AI Companion not initialized"}
    return {"status": "ok"}

# Main entry point for running the API server
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # 使用localhost而不是0.0.0.0可以直接在浏览器中打开
    uvicorn.run(app, host="localhost", port=8000) 