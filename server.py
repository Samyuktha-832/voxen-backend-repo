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
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import listen_and_transcribe
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
# ----------------------------
# Model-to-Provider Mapping (ALL USE SAME OLLAMA INSTANCE)
# ----------------------------
# ----------------------------
# Model-to-Provider Mapping (ALL USE SAME OLLAMA INSTANCE)
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

# Allow frontend origins (env: ALLOWED_ORIGINS="http://site1,https://site2")
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Default wide-open for connectivity; tighten in production by setting ALLOWED_ORIGINS
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Models
# ----------------------------
class ChatIn(BaseModel):
    message: str
    model: str = None  # Optional model parameter

class ChatOut(BaseModel):
    reply: str
    audio_base64: str

# ----------------------------
# AI Functions using LLMClient
# ----------------------------
def ask_ai(prompt: str, model: str = None) -> str:
    """
    Get AI response with optional model override
    
    Args:
        prompt: User's message
        model: Optional model name to use (overrides default)
    """
    try:
        # Configurable per-call timeout for LLM calls (seconds)
        llm_call_timeout = int(os.getenv("LLM_CALL_TIMEOUT_S", "180"))

        def run_with_timeout(fn, timeout_s):
            """Run fn() in a thread and return result or raise TimeoutError."""
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                try:
                    return fut.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                    raise TimeoutError(f"LLM call timed out after {timeout_s}s")

        messages = [
            {"role": "system", "content": "You are Voxen AI, a helpful and intelligent assistant."},
            {"role": "user", "content": prompt}
        ]

        # If a specific model is requested, determine the correct provider
        if model and model in MODEL_TO_PROVIDER:
            target_provider = MODEL_TO_PROVIDER[model]
            print(f"🎯 Model '{model}' maps to provider: {target_provider}")
            
            try:
                # Create a new client for this specific provider
                model_client = LLMClient(provider_name=target_provider, config_path="config.json")
                print(f"🔹 Using provider: {target_provider} | model: {model}")
                
                reply = run_with_timeout(
                    lambda: model_client.chat(messages=messages, model=model), 
                    llm_call_timeout
                )
                print(f"🔍 Raw reply from {target_provider}:{model}: {repr(reply)}")
                if reply and isinstance(reply, str) and reply.strip():
                    return reply.strip()
                    
            except TimeoutError as te:
                print(f"⏱ Model {model} timed out: {te}")
                raise
            except Exception as e:
                print(f"❌ Model {model} failed: {e}")
                # Don't raise - fall through to default behavior
        
        # Default behavior: use the primary provider
        print(f"🔹 Using primary provider: {llm_client.provider_name} | model: {llm_client.model}")
        
        # If primary is Ollama and no specific model selected, try small models
        if llm_client.provider_name.startswith("ollama") and not model:
            small_models = ["qwen2.5:0.5b", "gemma3:1b", llm_client.model]
            for m in small_models:
                try:
                    print(f"🔸 Trying Ollama model: {m}")
                    reply = run_with_timeout(
                        lambda: llm_client.chat(messages=messages, model=m), 
                        llm_call_timeout
                    )
                    print(f"🔍 Raw reply from {llm_client.provider_name}:{m}: {repr(reply)}")
                    if reply and isinstance(reply, str) and reply.strip():
                        return reply.strip()
                except TimeoutError as te:
                    print(f"⏱ Ollama model {m} timed out: {te}")
                except Exception as e:
                    print(f"⚠ Ollama model {m} failed: {e}")
            # If all models failed, fall through to fallback
            print("⚠ All Ollama models failed, trying fallback...")
        else:
            # Try the primary provider once
            try:
                target_model = model if model else llm_client.model
                reply = run_with_timeout(
                    lambda: llm_client.chat(messages=messages, model=target_model), 
                    llm_call_timeout
                )
                print(f"🔍 Raw reply from {llm_client.provider_name}: {repr(reply)}")
                if reply and isinstance(reply, str) and reply.strip():
                    return reply.strip()
            except TimeoutError as te:
                print(f"⏱ Primary provider call timed out: {te}")
            except Exception as e:
                print(f"⚠ Primary provider call failed: {e}")

        # Fallback logic
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            fallback_provider = config.get("fallback_provider")
        except Exception:
            fallback_provider = None

        if fallback_provider:
            try:
                print(f"🔄 Falling back to provider: {fallback_provider}")
                fallback_client = LLMClient(provider_name=fallback_provider)
                try:
                    reply = run_with_timeout(
                        lambda: fallback_client.chat(messages=messages), 
                        llm_call_timeout
                    )
                    print(f"🔍 Raw reply from {fallback_provider}: {repr(reply)}")
                    if reply and isinstance(reply, str) and reply.strip():
                        return reply.strip()
                except TimeoutError as te:
                    print(f"⏱ Fallback provider call timed out: {te}")
            except Exception as e:
                print(f"❌ Fallback provider also failed: {e}")

        return "I'm having trouble connecting to my AI service right now."

    except Exception as e:
        print(f"⚠ AI Error: {e}")
        return "I'm sorry, I could not process that request right now."

# ----------------------------
# API Routes
# ----------------------------
@app.post("/api/chat", response_model=ChatOut)
def chat(req: ChatIn):
    try:
        start_time = time.time()
        print(f"📩 User: {req.message}  [chat start @ {start_time:.3f}]")
        
        # Get selected model from request (if provided)
        selected_model = getattr(req, 'model', None)
        print(f"🎯 Selected model: {selected_model or 'default'}")

        t_llm_start = time.time()
        reply = ask_ai(req.message, model=selected_model)
        t_llm_end = time.time()
        print(f"🔁 ask_ai finished (len={len(reply) if reply else 0}) in {t_llm_end - t_llm_start:.2f}s")

        # Ensure reply is never empty
        if not reply or not reply.strip():
            reply = "I'm sorry, I didn't get that."

        print(f"✅ Assistant: {reply}")

        # Use temporary file for TTS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            filename = tmpfile.name
            tmpfile.close()

            try:
                t_tts_start = time.time()
                print(f"🔊 Starting TTS -> {filename} [@ {t_tts_start:.3f}]")
                engine.save_to_file(reply, filename)
                engine.runAndWait()
                t_tts_end = time.time()
                print(f"🔊 TTS complete in {t_tts_end - t_tts_start:.2f}s")
            except Exception as e:
                print(f"⚠ TTS Error: {e}")
                # fallback: generate silent audio if TTS fails
                with open(filename, "wb") as f:
                    f.write(b"")

            # Convert audio to base64 (and measure)
            t_enc_start = time.time()
            with open(filename, "rb") as f:
                data = f.read()
                audio_base64 = base64.b64encode(data).decode("utf-8")
            t_enc_end = time.time()
            try:
                file_size = len(data)
            except Exception:
                file_size = 0
            print(f"📦 Audio encoded: {file_size} bytes, encoding took {t_enc_end - t_enc_start:.2f}s, total chat time {t_enc_end - start_time:.2f}s")

        return JSONResponse(content={
            "reply": reply, 
            "audio_base64": audio_base64,
            "model_used": selected_model or llm_client.model
        })

    except Exception as e:
        print(f"⚠ Chat Endpoint Error: {e}")
        # Always return fallback
        fallback_reply = "I'm sorry, I didn't get that."
        return JSONResponse(content={"reply": fallback_reply, "audio_base64": ""})


@app.get("/api/models")
def get_available_models():
    """Get list of available Ollama models"""
    try:
        # Query Ollama directly for available models
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            data = response.json()
            models_list = []
            
            # Only these 3 models we support
            supported_models = {
                "qwen2.5:0.5b": "Qwen 2.5 (0.5B) - Ultra Fast ⚡",
                "tinyllama:1.1b": "TinyLlama (1.1B) - Compact 🔷",
                "gemma3:1b": "gemma3:1b - Efficient 🧠"
            }
            
            # Only return models that are actually installed
            for model_info in data.get("models", []):
                model_name = model_info.get("name")
                if model_name in supported_models:
                    models_list.append({
                        "name": model_name,
                        "display_name": supported_models[model_name],
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
        # Fallback to default model only
        return {"models": [
            {
                "name": "qwen2.5:0.5b", 
                "display_name": "Qwen 2.5 (0.5B) - Ultra Fast ⚡",
                "provider": "ollama"
            }
        ]}