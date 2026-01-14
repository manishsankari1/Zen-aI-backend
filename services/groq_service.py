import os
from typing import List, Optional
from groq import Groq
import asyncio

class GroqService:
    """
    Service for interacting with Groq API
    """
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.timeout = 60.0
    
    async def check_connection(self) -> bool:
        """
        Check if Groq API is accessible
        """
        try:
            # Run the synchronous Groq call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
            )
            return True
        except Exception as e:
            print(f"Groq connection check failed: {e}")
            return False
    
    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """
        Generate response from Groq API
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
            
            # Call Groq API in a thread pool (since Groq SDK is synchronous)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=500
                )
            )
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise Exception("No response from Groq API")
                    
        except Exception as e:
            print(f"Groq API error: {e}")
            # Fallback response if Groq fails
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """
        Fallback response when Groq is unavailable
        """
        return """I hear you.

Right now, I'm having trouble connecting to my awareness engine, but I want you to know that what you're feeling is valid.

Try this simple practice:
1. Close your eyes for a moment
2. Take three slow breaths
3. Notice what you're feeling without judging it
4. Just be with it

Sometimes the most powerful thing is simply to observe."""
