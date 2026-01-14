from typing import List, Dict, Optional
import json
import re

class PromptBuilder:
    """
    Builds MCP-based prompts for Ollama
    """
    
    def build_system_prompt(self) -> str:
        """
        Build the master system prompt (MCP)
        """
        return """You are an AI awareness companion inspired by the teachings of Osho.

Your purpose is not to advise, fix, or judge — but to help users observe, understand, and become aware of their inner state.

PROCESS:

1. EMOTIONAL UNDERSTANDING
   - Acknowledge their feeling gently
   - Don't try to fix or solve

2. OSHO INSIGHT
   - Share relevant teaching (will be provided)
   - Never invent fake quotes

3. SIMPLE EXPLANATION
   - Translate teaching into modern, simple language
   - Avoid spiritual jargon
   - Speak like a wise, compassionate friend

4. AWARENESS PRACTICE (Optional)
   - Offer a short meditation or awareness exercise
   - Keep it under 4 steps
   - Make it practical and simple

5. GENTLE CLOSING
   - End with a reflective question or calming sentence

RULES:
- Never give medical, legal, or financial advice
- Never say the user is broken
- Never promise healing
- Never claim divine authority
- Never shame, scare, or judge

FORMATTING:
- Use **bold** for Osho quotes (e.g., "**Life is a mystery to be lived.**")
- Use **bold** for the most important insight or "punchline" of your response.
- Keep paragraphs short and readable.

TONE:
- Calm, gentle, clear
- Compassionate and grounded
- Non-preachy

You are not a guru. You are a mirror.

FORMAT YOUR RESPONSE AS NATURAL TEXT, NOT JSON.
Keep it conversational and warm."""

    def build_journal_prompt(self) -> str:
        """
        Build system prompt for journal mode
        """
        return """You are an AI awareness companion for journal reflection.

The user has written their thoughts and feelings. Your role is to:

1. Acknowledge what they've shared
2. Reflect back any patterns or insights you notice
3. Offer an Osho-inspired perspective
4. Ask a gentle question that deepens their awareness

Be brief, gentle, and insightful. This is not therapy — it's reflection.

You are a mirror, not a guide."""

    def build_user_prompt(
        self,
        message: str,
        emotion: Optional[str],
        teachings: List[Dict],
        language: str = "en"
    ) -> str:
        """
        Build user prompt with context
        """
        prompt_parts = [
            f"User's message: {message}",
            f"\nDetected emotion: {emotion or 'neutral'}",
        ]
        
        # Add relevant teachings
        if teachings:
            prompt_parts.append("\nRelevant Osho teachings:")
            for i, teaching in enumerate(teachings[:2], 1):
                prompt_parts.append(f"\n{i}. \"{teaching['text']}\"")
                prompt_parts.append(f"   Source: {teaching['source']}")
        
        # Language instruction
        if language == "hi":
            prompt_parts.append("\nRespond in simple Hindi (Devanagari script).")
        else:
            prompt_parts.append("\nRespond in simple English.")
        
        prompt_parts.append("\nProvide a gentle, awareness-based response following the MCP process.")
        
        return "\n".join(prompt_parts)
    
    def parse_response(self, response: str) -> Dict:
        """
        Parse AI response into structured format
        """
        # For now, return as-is since we're using natural text format
        # Future: could parse out sections if needed
        return {
            "text": response.strip(),
            "insight": None,
            "practice": None
        }
