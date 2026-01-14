"""
Test script to verify Groq API integration
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.groq_service import GroqService
import asyncio

async def test_groq():
    print("🧪 Testing Groq API Integration...")
    api_key = os.getenv('GROQ_API_KEY', '')
    masked_key = f"{api_key[:10]}{'*' * 20}..." if api_key else "NOT SET"
    print(f"API Key: {masked_key}")
    print(f"Model: {os.getenv('GROQ_MODEL')}")
    print()
    
    try:
        # Initialize service
        groq_service = GroqService()
        print("✅ Groq service initialized successfully")
        
        # Test connection
        print("\n🔌 Testing connection...")
        is_connected = await groq_service.check_connection()
        
        if is_connected:
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed")
            return
        
        # Test generation
        print("\n💬 Testing response generation...")
        system_prompt = "You are a compassionate awareness companion inspired by Osho's teachings."
        user_prompt = "I'm feeling anxious about the future."
        
        response = await groq_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        print("\n📝 Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_groq())
