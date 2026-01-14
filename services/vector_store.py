import chromadb
from chromadb.config import Settings
import os
# Disable telemetry aggressively
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from typing import List, Dict, Optional

class OshoVectorStore:
    """
    Vector database for storing and retrieving Osho teachings
    """
    
    def __init__(self):
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("osho_teachings")
        except:
            self.collection = self.client.create_collection(
                name="osho_teachings",
                metadata={"description": "Osho quotes and teachings"}
            )
            self._initialize_teachings()
    
    def _initialize_teachings(self):
        """
        Initialize database with core Osho teachings
        """
        teachings = [
            # Anxiety & Worry
            {
                "text": "Anxiety is the gap between the now and the then. If you are in the now, there is no anxiety.",
                "emotion": "anxiety",
                "source": "The Book of Secrets",
                "theme": "present_moment"
            },
            {
                "text": "Worry is a futile thing. It has never helped anybody. It only destroys your today.",
                "emotion": "anxiety",
                "source": "Inspired by Osho",
                "theme": "worry"
            },
            
            # Sadness & Grief
            {
                "text": "Sadness gives depth. Happiness gives height. Sadness gives roots. Happiness gives branches.",
                "emotion": "sadness",
                "source": "Everyday Osho",
                "theme": "acceptance"
            },
            {
                "text": "When you are sad, be sad. Don't try to escape from it. Let it be there. Watch it.",
                "emotion": "sadness",
                "source": "Inspired by Osho",
                "theme": "witnessing"
            },
            
            # Confusion & Uncertainty
            {
                "text": "When you don't know, you are open. When you think you know, you are closed.",
                "emotion": "confusion",
                "source": "The Book of Understanding",
                "theme": "not_knowing"
            },
            {
                "text": "Confusion is the beginning of clarity. Stay with it.",
                "emotion": "confusion",
                "source": "Inspired by Osho",
                "theme": "acceptance"
            },
            
            # Loneliness
            {
                "text": "Loneliness is the absence of the other. Aloneness is the presence of oneself.",
                "emotion": "loneliness",
                "source": "The Path of Meditation",
                "theme": "aloneness"
            },
            {
                "text": "You are born alone, you will die alone. And in between, you are alone. Accept it.",
                "emotion": "loneliness",
                "source": "Inspired by Osho",
                "theme": "acceptance"
            },
            
            # Anger
            {
                "text": "Anger is beautiful. Watch it. Don't act on it, don't suppress it. Just watch.",
                "emotion": "anger",
                "source": "Tantra: The Supreme Understanding",
                "theme": "witnessing"
            },
            {
                "text": "When anger comes, close your eyes and watch where it is in the body. Watch it like a scientist.",
                "emotion": "anger",
                "source": "Inspired by Osho",
                "theme": "awareness"
            },
            
            # Meaninglessness
            {
                "text": "Life has no meaning. And that is its beauty. You are free to create your own meaning.",
                "emotion": "meaninglessness",
                "source": "The Book of Understanding",
                "theme": "freedom"
            },
            
            # Overthinking
            {
                "text": "The mind is a beautiful servant but a dangerous master.",
                "emotion": "overthinking",
                "source": "Inspired by Osho",
                "theme": "mind"
            },
            {
                "text": "Thoughts are like clouds. Watch them come and go. You are the sky.",
                "emotion": "overthinking",
                "source": "Meditation: The First and Last Freedom",
                "theme": "witnessing"
            },
            
            # General Awareness
            {
                "text": "Be — don't try to become.",
                "emotion": "neutral",
                "source": "The Book of Secrets",
                "theme": "being"
            },
            {
                "text": "Awareness is the greatest alchemy there is. Just go on becoming more and more aware.",
                "emotion": "neutral",
                "source": "The Path of Meditation",
                "theme": "awareness"
            },
            {
                "text": "The moment you become aware of a feeling, it starts changing.",
                "emotion": "neutral",
                "source": "Inspired by Osho",
                "theme": "transformation"
            }
        ]
        
        # Add teachings to vector store
        for i, teaching in enumerate(teachings):
            self.collection.add(
                documents=[teaching["text"]],
                metadatas=[{
                    "emotion": teaching["emotion"],
                    "source": teaching["source"],
                    "theme": teaching["theme"]
                }],
                ids=[f"teaching_{i}"]
            )
    
    def search(self, query: str, emotion: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant teachings based on query and emotion
        """
        try:
            # Build where filter for emotion
            where_filter = None
            if emotion and emotion != "neutral":
                where_filter = {"emotion": emotion}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
            teachings = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    teachings.append({
                        "text": results["documents"][0][i],
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                        "theme": results["metadatas"][0][i].get("theme", "general")
                    })
            
            return teachings
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def is_ready(self) -> bool:
        """
        Check if vector store is ready
        """
        try:
            return self.collection.count() > 0
        except:
            return False
