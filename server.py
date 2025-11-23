import whisper
import pyttsx3
import speech_recognition as sr
import json
import atexit
import requests
import uuid
import tempfile
import base64
import time
import concurrent.futures
from fastapi import FastAPI, Request, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from llm_client import LLMClient
import os

os.environ['OLLAMA_NUM_GPU'] = '0'

# ----------------------------
# Setup
# ----------------------------
print("🔄 Loading Whisper model (base)...")
model = whisper.load_model("base")

# Global TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

def cleanup_tts():
    try:
        if 'engine' in globals() and engine is not None:
            engine.stop()
            del globals()['engine']
    except Exception:
        pass

atexit.register(cleanup_tts)

recognizer = sr.Recognizer()
mic = sr.Microphone()

# ----------------------------
# Model-to-Provider Mapping
# ----------------------------
MODEL_TO_PROVIDER = {
    "qwen2.5:0.5b": "ollama",
    "tinyllama:1.1b": "ollama",
    "gemma3:1b": "ollama",
}

# ----------------------------
# Setup LLM Client
# ----------------------------
try:
    llm_client = LLMClient(provider_name=None, config_path="config.json")
    print(f"✅ Server connected to: {llm_client.provider_name} | model: {llm_client.model}")
except Exception as e:
    print(f"❌ Failed to initialize LLM client: {e}")
    raise RuntimeError(f"❌ Could not connect to AI provider: {e}")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Models
# ----------------------------
class ChatIn(BaseModel):
    message: str
    model: Optional[str] = None
    provider: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    audio_base64: str
    model_used: Optional[str] = None

class ConfigUpdate(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None

class ConnectionTest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None

# ----------------------------
# AI Functions
# ----------------------------
def ask_ai_with_provider(prompt: str, provider: str = None, api_key: str = None, model: str = None) -> str:
    """
    Get AI response with dynamic provider switching
    
    Args:
        prompt: User's message
        provider: Provider to use (ollama, openai, anthropic, openrouter)
        api_key: API key for external providers
        model: Model name to use
    """
    try:
        llm_call_timeout = int(os.getenv("LLM_CALL_TIMEOUT_S", "180"))

        def run_with_timeout(fn, timeout_s):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                try:
                    return fut.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    fut.cancel()
                    raise TimeoutError(f"LLM call timed out after {timeout_s}s")

        messages = [
            {"role": "system", "content": "You are Voxen AI, a helpful and intelligent assistant."},
            {"role": "user", "content": prompt}
        ]

        # If external provider (not ollama), create temporary client with API key
        if provider and provider != "ollama" and api_key:
            print(f"🔑 Using external provider: {provider}")
            
            # Create temporary config
            temp_config = {
                "providers": {
                    provider: {
                        "api_key": api_key,
                        "default_model": model
                    }
                },
                "default_provider": provider
            }
            
            # Add base URLs for external providers
            if provider == "openai":
                temp_config["providers"][provider]["base_url"] = "https://api.openai.com/v1"
            elif provider == "anthropic":
                temp_config["providers"][provider]["base_url"] = "https://api.anthropic.com/v1"
            elif provider == "openrouter":
                temp_config["providers"][provider]["base_url"] = "https://openrouter.ai/api/v1"
            
            temp_config_path = f"temp_config_{uuid.uuid4().hex}.json"
            
            try:
                with open(temp_config_path, "w") as f:
                    json.dump(temp_config, f)
                
                temp_client = LLMClient(provider_name=provider, config_path=temp_config_path)
                
                reply = run_with_timeout(
                    lambda: temp_client.chat(messages=messages, model=model),
                    llm_call_timeout
                )
                
                if reply and isinstance(reply, str) and reply.strip():
                    return reply.strip()
                    
            finally:
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        
        # Use Ollama or default provider
        print(f"🔹 Using Ollama provider")
        
        # Determine which model to use
        target_model = model if model else llm_client.model
        
        # If specific model requested and it's mapped to a provider
        if model and model in MODEL_TO_PROVIDER:
            target_provider = MODEL_TO_PROVIDER[model]
            print(f"🎯 Model '{model}' maps to provider: {target_provider}")
        
        # Try the requested model or default
        try:
            print(f"🔸 Trying model: {target_model}")
            reply = run_with_timeout(
                lambda: llm_client.chat(messages=messages, model=target_model),
                llm_call_timeout
            )
            
            if reply and isinstance(reply, str) and reply.strip():
                return reply.strip()
        except Exception as e:
            print(f"⚠️ Model {target_model} failed: {e}")
        
        # Fallback to small models if Ollama
        if llm_client.provider_name.startswith("ollama"):
            small_models = ["qwen2.5:0.5b", "gemma3:1b", "tinyllama:1.1b"]
            for m in small_models:
                try:
                    print(f"🔸 Trying fallback model: {m}")
                    reply = run_with_timeout(
                        lambda: llm_client.chat(messages=messages, model=m),
                        llm_call_timeout
                    )
                    if reply and isinstance(reply, str) and reply.strip():
                        return reply.strip()
                except Exception as e:
                    print(f"⚠️ Fallback model {m} failed: {e}")
                    continue
        
        return "I'm having trouble connecting to my AI service right now."

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return "I'm sorry, I could not process that request right now."

# ----------------------------
# API Routes
# ----------------------------
@app.post("/api/chat", response_model=ChatOut)
async def chat(
    req: ChatIn,
    x_api_key: Optional[str] = Header(None),
    x_provider: Optional[str] = Header(None)
):
    """
    Main chat endpoint - supports both Ollama and external API providers
    """
    try:
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"📩 User: {req.message}")
        print(f"🔧 Provider: {x_provider or req.provider or 'ollama'}")
        print(f"🎯 Model: {req.model or 'default'}")
        print(f"🔑 API Key: {'***' + x_api_key[-4:] if x_api_key else 'None'}")
        print(f"{'='*60}")
        
        # Determine provider
        provider = x_provider or req.provider or "ollama"
        api_key = x_api_key
        selected_model = req.model
        
        # Get AI response
        t_llm_start = time.time()
        reply = ask_ai_with_provider(
            prompt=req.message,
            provider=provider,
            api_key=api_key,
            model=selected_model
        )
        t_llm_end = time.time()
        
        print(f"✅ AI Response received in {t_llm_end - t_llm_start:.2f}s")
        print(f"💬 Reply: {reply[:100]}{'...' if len(reply) > 100 else ''}")

        # Ensure reply is never empty
        if not reply or not reply.strip():
            reply = "I'm sorry, I didn't get that."

        # Generate TTS audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            filename = tmpfile.name
            tmpfile.close()

            try:
                t_tts_start = time.time()
                engine.save_to_file(reply, filename)
                engine.runAndWait()
                t_tts_end = time.time()
                print(f"🔊 TTS generated in {t_tts_end - t_tts_start:.2f}s")
            except Exception as e:
                print(f"⚠️ TTS Error: {e}")
                with open(filename, "wb") as f:
                    f.write(b"")

            # Convert audio to base64
            with open(filename, "rb") as f:
                data = f.read()
                audio_base64 = base64.b64encode(data).decode("utf-8")
            
            print(f"📦 Audio encoded: {len(data)} bytes")
            print(f"⏱️ Total request time: {time.time() - start_time:.2f}s\n")

            # Clean up temp file
            try:
                os.remove(filename)
            except Exception:
                pass

        return JSONResponse(content={
            "reply": reply,
            "audio_base64": audio_base64,
            "model_used": selected_model or llm_client.model,
            "provider_used": provider
        })

    except Exception as e:
        print(f"❌ Chat Endpoint Error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "reply": "I'm sorry, I encountered an error. Please try again.",
                "audio_base64": "",
                "error": str(e)
            }
        )


@app.get("/api/models")
def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            data = response.json()
            models_list = []
            
            # Map model names to display names
            for model_info in data.get("models", []):
                model_name = model_info.get("name", "")
                
                models_list.append({
                    "name": model_name,
                    "provider": "ollama",
                    "size": model_info.get("size", 0)
                })
            
            # Sort by size (smallest first)
            models_list.sort(key=lambda x: x.get("size", 0))
            
            print(f"✅ Returning {len(models_list)} available models")
            return {"models": models_list}
        else:
            raise Exception("Ollama not responding")
            
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        # Return default models as fallback
        return {"models": [
            {"name": "qwen2.5:0.5b", "provider": "ollama"},
            {"name": "tinyllama:1.1b", "provider": "ollama"},
            {"name": "gemma3:1b", "provider": "ollama"}
        ]}


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update AI provider configuration"""
    try:
        with open("config.json", "r") as f:
            full_config = json.load(f)
        
        # Initialize providers if not exists
        if "providers" not in full_config:
            full_config["providers"] = {}
        
        # Update configuration based on provider
        if config.provider == "ollama":
            full_config["default_provider"] = "ollama"
            full_config["fallback_provider"] = None
            
        else:  # External provider (openai, anthropic, openrouter)
            if not config.api_key:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"API key required for {config.provider}"}
                )
            
            # Create provider config if not exists
            if config.provider not in full_config["providers"]:
                full_config["providers"][config.provider] = {}
            
            # Update provider settings
            full_config["providers"][config.provider]["api_key"] = config.api_key
            full_config["providers"][config.provider]["default_model"] = config.model
            
            # Set as default provider with Ollama fallback
            full_config["default_provider"] = config.provider
            full_config["fallback_provider"] = "ollama"
        
        # Save updated config
        with open("config.json", "w") as f:
            json.dump(full_config, f, indent=2)
        
        # Reinitialize LLM client
        global llm_client
        llm_client = LLMClient(provider_name=config.provider, config_path="config.json")
        
        print(f"✅ Configuration updated: {config.provider}")
        
        return {
            "success": True,
            "message": f"Configuration updated to use {config.provider}",
            "provider": config.provider,
            "fallback": full_config.get("fallback_provider")
        }
        
    except Exception as e:
        print(f"❌ Config update error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/test-connection")
async def test_connection(test: ConnectionTest):
    """Test connection to AI provider"""
    try:
        print(f"🔍 Testing connection to {test.provider}...")
        
        if test.provider == "ollama":
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.ok:
                models_count = len(response.json().get("models", []))
                return {
                    "success": True,
                    "message": f"Ollama is running with {models_count} models available",
                    "models": models_count
                }
            else:
                raise Exception("Ollama is not responding")
        
        elif test.provider in ["openai", "anthropic", "openrouter"]:
            if not test.api_key:
                raise Exception("API key required")
            
            # Create temporary config
            temp_config = {
                "providers": {
                    test.provider: {
                        "api_key": test.api_key,
                        "default_model": test.model
                    }
                },
                "default_provider": test.provider
            }
            
            # Add base URLs
            if test.provider == "openai":
                temp_config["providers"][test.provider]["base_url"] = "https://api.openai.com/v1"
            elif test.provider == "anthropic":
                temp_config["providers"][test.provider]["base_url"] = "https://api.anthropic.com/v1"
            elif test.provider == "openrouter":
                temp_config["providers"][test.provider]["base_url"] = "https://openrouter.ai/api/v1"
            
            temp_config_path = f"temp_test_{uuid.uuid4().hex}.json"
            
            try:
                with open(temp_config_path, "w") as f:
                    json.dump(temp_config, f)
                
                test_client = LLMClient(provider_name=test.provider, config_path=temp_config_path)
                
                # Send test message
                test_messages = [{"role": "user", "content": "Hi"}]
                response = test_client.chat(messages=test_messages, model=test.model)
                
                if response:
                    return {
                        "success": True,
                        "message": f"Successfully connected to {test.provider}",
                        "provider": test.provider,
                        "model": test.model
                    }
                else:
                    raise Exception("No response from API")
                    
            finally:
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        
        else:
            raise Exception(f"Unknown provider: {test.provider}")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": f"Failed to connect to {test.provider}: {str(e)}"
            }
        )


@app.get("/api/config")
async def get_config():
    """Get current configuration (without exposing API keys)"""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        safe_config = {
            "default_provider": config.get("default_provider"),
            "fallback_provider": config.get("fallback_provider"),
            "providers": {}
        }
        
        for provider, settings in config.get("providers", {}).items():
            safe_config["providers"][provider] = {
                "configured": bool(settings.get("api_key")),
                "model": settings.get("default_model")
            }
        
        return safe_config
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "provider": llm_client.provider_name,
        "model": llm_client.model
    }


# Run server
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting VOXEN AI Server on http://localhost:8000")
    print("📡 CORS enabled for all origins")
    print(f"🤖 Default provider: {llm_client.provider_name}")
    print(f"🎯 Default model: {llm_client.model}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
