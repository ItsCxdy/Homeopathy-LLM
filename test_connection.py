import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_available_models():
    """Get list of available models from OpenRouter"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
        return []
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            models = response.json().get('data', [])
            return models
        else:
            print(f"❌ Could not fetch models list (Status: {response.status_code})")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching models: {str(e)}")
        return []

def test_model_connection(model_id):
    """Test connection with a specific model"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "Homeopathy AI Doctor Test"
    }
    
    test_data = {
        "model": model_id,
        "messages": [
            {
                "role": "user", 
                "content": "Hello! Please respond with just 'OK' to confirm connection."
            }
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result["choices"][0]["message"]["content"]
            return True, f"✅ {model_id} - SUCCESS: {message.strip()}"
        else:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            return False, f"❌ {model_id} - FAILED: {error_msg}"
            
    except Exception as e:
        return False, f"❌ {model_id} - ERROR: {str(e)}"

def main():
    print("🚀 OpenRouter Connection Test - Finding Available Models")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
        return
    
    print(f"✅ API Key found (starts with): {api_key[:10]}...")
    
    # Get available models
    print("\n🔍 Fetching available models...")
    models = get_available_models()
    
    if not models:
        print("❌ Could not fetch models list")
        return
    
    print(f"✅ Found {len(models)} total models")
    
    # Filter for free models that might work for us
    free_models = [
        # DeepSeek models
        "deepseek/deepseek-chat",
        "deepseek/deepseek-coder",
        "deepseek/deepseek-llm-67b-chat",
        "deepseek/deepseek-llm-7b-chat",
        
        # Other free models
        "google/gemma-7b-it:free",
        "microsoft/wizardlm-2-8x22b:free", 
        "meta-llama/llama-3-8b-instruct:free",
        "meta-llama/llama-3-70b-instruct:free",
        "qwen/qwen-2-7b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        
        # Try without :free suffix
        "deepseek/deepseek-chat",
        "google/gemma-7b-it",
        "meta-llama/llama-3-8b-instruct",
    ]
    
    print("\n🧪 Testing potential free models...")
    print("-" * 60)
    
    working_models = []
    
    for model_id in free_models:
        success, message = test_model_connection(model_id)
        print(message)
        if success:
            working_models.append(model_id)
    
    print("\n" + "=" * 60)
    if working_models:
        print("🎉 WORKING MODELS FOUND:")
        for model in working_models:
            print(f"   ✅ {model}")
        print(f"\n💡 Use one of these models in your doctor_bot.py")
    else:
        print("💡 No free models worked. Try these solutions:")
        print("1. Check your OpenRouter account at https://openrouter.ai/")
        print("2. Some models might require sign-in or have usage limits")
        print("3. Try models without the ':free' suffix")
        print("4. Consider using a different API provider")

if __name__ == "__main__":
    main()