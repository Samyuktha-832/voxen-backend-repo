import whisper
import pyttsx3
import speech_recognition as sr
import os
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

class ChatOut(BaseModel):
    reply: str
    audio_base64: str

# ----------------------------
# AI Functions using LLMClient
# ----------------------------
def ask_ai(prompt: str) -> str:
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

        # First, try the already-initialized global llm_client (uses default_provider from config.json)
        try:
            print(f"🔹 Trying primary provider (singleton): {llm_client.provider_name} | model: {llm_client.model}")

            # If primary is Ollama, prefer smaller local models first to avoid memory/GPU issues
            if llm_client.provider_name == "ollama":
                small_models = ["qwen3:0.6b", "phi3", llm_client.model]
                for m in small_models:
                    try:
                        print(f"🔸 Trying Ollama model: {m}")
                        reply = run_with_timeout(lambda: llm_client.chat(messages=messages, model=m), llm_call_timeout)
                        print(f"🔍 Raw reply from ollama:{m}: {repr(reply)}")
                        if reply and isinstance(reply, str) and reply.strip():
                            return reply.strip()
                    except TimeoutError as te:
                        print(f"⏱ Ollama model {m} timed out: {te}")
                    except Exception as e:
                        print(f"⚠ Ollama model {m} failed: {e}")
                # if none of the Ollama models worked, raise to trigger fallback
                raise RuntimeError("All tested Ollama models failed or returned invalid responses")
            else:
                try:
                    reply = run_with_timeout(lambda: llm_client.chat(messages=messages), llm_call_timeout)
                    print(f"🔍 Raw reply from {llm_client.provider_name}: {repr(reply)}")
                    if reply and isinstance(reply, str) and reply.strip():
                        return reply.strip()
                except TimeoutError as te:
                    print(f"⏱ Primary provider call timed out: {te}")
                except Exception as e:
                    print(f"⚠ Primary provider call failed: {e}")
        except Exception as e:
            print(f"❌ Primary provider (singleton) failed: {e}")

        # If primary failed, read config and attempt fallback provider if configured
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
                    reply = run_with_timeout(lambda: fallback_client.chat(messages=messages), llm_call_timeout)
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

        t_llm_start = time.time()
        reply = ask_ai(req.message)
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

        return JSONResponse(content={"reply": reply, "audio_base64": audio_base64})

    except Exception as e:
        print(f"⚠ Chat Endpoint Error: {e}")
        # Always return fallback
        fallback_reply = "I'm sorry, I didn't get that."
        return JSONResponse(content={"reply": fallback_reply, "audio_base64": ""})




@app.get("/api/listen")
def listen_endpoint():
    try:
        text = listen_and_transcribe()  # calls main.py microphone function
        if not text:
            text = ""  # Ensure we return empty string, not None
        return JSONResponse(content={"message": text})
    except Exception as e:
        print(f"⚠ Listen Endpoint Error: {e}")
        return JSONResponse(content={"message": ""}, status_code=500)


@app.get("/api/profile")
def profile():
    return {
        "user": {
            "id": "123",
            "username": "shruti",
            "full_name": "Shruti S Sajeev",
            "profile_picture": ""  # You can put a URL if available
        }
    }