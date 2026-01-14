import httpx
import os
from typing import List, Optional

class OllamaService:
    """
    Service for interacting with local Ollama instance
    """
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3:8b")
        self.timeout = 60.0
    
    async def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """
        Generate response from Ollama
        """
        try:
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-6:])  # Last 3 exchanges
            
            # Add current user message
            messages.append({"role": "user", "content": user_prompt})
            
            # Call Ollama API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 500
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            # Fallback response if Ollama fails
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """
        Fallback response when Ollama is unavailable
        """
        return """I hear you.

Right now, I'm having trouble connecting to my awareness engine, but I want you to know that what you're feeling is valid.

Try this simple practice:
1. Close your eyes for a moment
2. Take three slow breaths
3. Notice what you're feeling without judging it
4. Just be with it

Sometimes the most powerful thing is simply to observe."""
