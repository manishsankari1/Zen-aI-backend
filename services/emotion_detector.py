import re
from typing import Optional

class EmotionDetector:
    """
    Detects emotional state from user input using keyword matching
    """
    
    def __init__(self):
        self.emotion_keywords = {
            "sadness": [
                "sad", "depressed", "down", "low", "unhappy", "miserable", 
                "heartbroken", "grief", "crying", "tears", "hurt"
            ],
            "anxiety": [
                "anxious", "worried", "stressed", "nervous", "tense", "panic", 
                "overwhelmed", "restless", "fear", "scared", "afraid"
            ],
            "confusion": [
                "confused", "lost", "unclear", "uncertain", "don't know", 
                "stuck", "directionless", "bewildered", "puzzled"
            ],
            "loneliness": [
                "lonely", "alone", "isolated", "disconnected", "empty", 
                "abandoned", "nobody", "no one"
            ],
            "anger": [
                "angry", "frustrated", "irritated", "mad", "furious", 
                "annoyed", "resentful", "rage", "pissed"
            ],
            "meaninglessness": [
                "meaningless", "purposeless", "empty", "void", "pointless", 
                "lost meaning", "no purpose", "what's the point"
            ],
            "overthinking": [
                "overthinking", "can't stop thinking", "mind racing", 
                "thoughts won't stop", "mental chatter", "ruminating"
            ],
            "peace": [
                "peaceful", "calm", "content", "serene", "tranquil", 
                "relaxed", "at ease"
            ],
            "curiosity": [
                "curious", "wondering", "interested", "question", 
                "what is", "how does", "why"
            ]
        }
    
    def detect(self, text: str) -> Optional[str]:
        """
        Detect primary emotion from text
        Returns emotion name or 'neutral' if none detected
        """
        text_lower = text.lower()
        
        # Count matches for each emotion
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return "neutral"
    
    def get_emotion_description(self, emotion: str) -> str:
        """
        Get a gentle description of the emotion
        """
        descriptions = {
            "sadness": "a feeling of sadness",
            "anxiety": "some anxiety or worry",
            "confusion": "confusion or uncertainty",
            "loneliness": "loneliness",
            "anger": "anger or frustration",
            "meaninglessness": "a sense of meaninglessness",
            "overthinking": "overthinking",
            "peace": "peace",
            "curiosity": "curiosity",
            "neutral": "what you're experiencing"
        }
        return descriptions.get(emotion, "what you're feeling")
