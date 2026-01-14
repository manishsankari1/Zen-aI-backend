from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
from dotenv import load_dotenv
from services.emotion_detector import EmotionDetector
from services.vector_store import OshoVectorStore
from services.groq_service import GroqService
from services.prompt_builder import PromptBuilder

# Load environment variables
load_dotenv()

app = FastAPI(title="OSHO AI - Awareness Companion")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
emotion_detector = EmotionDetector()
vector_store = OshoVectorStore()
groq_service = GroqService()
prompt_builder = PromptBuilder()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    emotion: Optional[str] = None
    insight: Optional[dict] = None
    practice: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    model: str
    vector_db_ready: bool

# Health Check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if all services are running"""
    groq_status = await groq_service.check_connection()
    
    return HealthResponse(
        status="healthy" if groq_status else "degraded",
        ollama_connected=groq_status,
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        vector_db_ready=vector_store.is_ready()
    )

# Main Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user input and returns awareness-based response
    """
    try:
        # Step 1: Detect emotion
        emotion = emotion_detector.detect(request.message)
        
        # Step 2: Retrieve relevant Osho teachings
        teachings = vector_store.search(request.message, emotion, top_k=3)
        
        # Step 3: Build MCP-based prompt
        system_prompt = prompt_builder.build_system_prompt()
        user_prompt = prompt_builder.build_user_prompt(
            message=request.message,
            emotion=emotion,
            teachings=teachings,
            language=request.language
        )
        
        # Step 4: Get response from Groq
        response = await groq_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            conversation_history=request.conversation_history
        )
        
        # Step 5: Parse and structure response
        parsed_response = prompt_builder.parse_response(response)
        
        return ChatResponse(
            response=parsed_response.get("text", response),
            emotion=emotion,
            insight=parsed_response.get("insight"),
            practice=parsed_response.get("practice")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Journal Endpoint
@app.post("/journal")
async def journal_reflection(request: ChatRequest):
    """
    Journal mode - provides deeper reflection on user's written thoughts
    """
    try:
        # Similar to chat but with journal-specific prompt
        emotion = emotion_detector.detect(request.message)
        teachings = vector_store.search(request.message, emotion, top_k=2)
        
        system_prompt = prompt_builder.build_journal_prompt()
        user_prompt = prompt_builder.build_user_prompt(
            message=request.message,
            emotion=emotion,
            teachings=teachings,
            language=request.language
        )
        
        response = await groq_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return {"reflection": response, "emotion": emotion}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing journal: {str(e)}")

# Get Meditation Practices
@app.get("/practices/{emotion}")
async def get_practices(emotion: str):
    """
    Get meditation/awareness practices for specific emotion
    """
    practices = {
        "anxiety": {
            "title": "Watching the Breath",
            "steps": [
                "Sit comfortably and close your eyes",
                "Notice your breath without changing it",
                "When anxiety comes, just watch it like a cloud",
                "Return to the breath gently"
            ]
        },
        "sadness": {
            "title": "Allowing the Feeling",
            "steps": [
                "Find a quiet space",
                "Let the sadness be there without fighting it",
                "Feel where it sits in your body",
                "Breathe into that space with kindness"
            ]
        },
        "anger": {
            "title": "Witnessing the Fire",
            "steps": [
                "Notice the anger without acting on it",
                "Feel the heat in your body",
                "Watch it like you're watching a storm",
                "Let it pass through without holding on"
            ]
        },
        "confusion": {
            "title": "Sitting with Not Knowing",
            "steps": [
                "Sit in silence for 5 minutes",
                "Don't try to find answers",
                "Just be with the confusion",
                "Notice the space between thoughts"
            ]
        }
    }
    
    return practices.get(emotion.lower(), {
        "title": "Simple Awareness",
        "steps": [
            "Close your eyes",
            "Notice what you're feeling",
            "Don't judge it",
            "Just observe"
        ]
    })

# Root endpoint
@app.get("/")
async def root():
    return {
        "app": "OSHO AI - Awareness Companion",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
