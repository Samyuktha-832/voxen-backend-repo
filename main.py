import whisper
import speech_recognition as sr
import pyttsx3
import os
import json
from llm_client import LLMClient

# ----------------------------
# Load Configuration
# ----------------------------
config_path = "config.json"
if not os.path.exists(config_path):
    raise RuntimeError(f"❌ Config file not found at {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)

provider = config.get("default_provider", "openrouter")
provider_config = config.get("providers", {}).get(provider)

if not provider_config:
    raise RuntimeError(f"❌ No provider settings found for '{provider}' in config.json")

PRIMARY_MODEL = provider_config.get("default_model", "mistralai/mistral-7b-instruct:free")
FALLBACK_MODEL = "mistralai/mistral-7b-instruct:free"  # Reliable fallback

# ----------------------------
# Setup Whisper
# ----------------------------
print("🔄 Loading Whisper model (base)...")
model = whisper.load_model("base")
print("✅ Whisper model loaded")

# ----------------------------
# Setup Speech Recognition
# ----------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# ----------------------------
# Setup AI Client (OpenRouter from config.json)
# ----------------------------
try:
    client = LLMClient(provider_name=None, config_path=config_path)
    print(f"✅ AI Client initialized with model: {PRIMARY_MODEL}")
except Exception as e:
    print(f"❌ Failed to initialize AI client: {e}")
    raise

# ----------------------------
# Speak Function (Windows SAPI5 compatible)
# ----------------------------
def speak(text):
    """
    Convert text to speech using pyttsx3
    Re-initializes engine each time to prevent freezing on Windows
    """
    if not text or not isinstance(text, str) or not text.strip():
        print("⚠️ No valid text to speak")
        return

    print(f"🗣️ Assistant says: {text}")

    try:
        # Re-init pyttsx3 each time to avoid engine freeze
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)

        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine  # Force cleanup
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")

# ----------------------------
# Listen & Transcribe
# ----------------------------
def listen_and_transcribe():
    """
    Listen to microphone input and transcribe using Whisper
    Returns the transcribed text
    """
    temp_audio_file = "temp.wav"
    
    try:
        with mic as source:
            print("🎤 Listening... (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

        # Save audio to temporary file
        with open(temp_audio_file, "wb") as f:
            f.write(audio.get_wav_data())

        print("🔎 Transcribing with Whisper...")
        result = model.transcribe(temp_audio_file, fp16=False)
        text = result["text"].strip()
        
        # Clean up temp file
        try:
            os.remove(temp_audio_file)
        except:
            pass
            
        print(f"👤 You said: {text}")
        return text
        
    except sr.WaitTimeoutError:
        print("⚠️ No speech detected (timeout)")
        return ""
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""

# ----------------------------
# Get AI Response with Fallback
# ----------------------------
def ask_ai(prompt):
    """
    Send prompt to AI and get response
    Uses primary model with automatic fallback to secondary model
    """
    if not prompt or not prompt.strip():
        return "I didn't hear anything. Could you please repeat that?"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant named ShrutiBot. Keep responses concise and natural."},
        {"role": "user", "content": prompt}
    ]

    # Try primary model
    try:
        print(f"🔹 Using primary model: {PRIMARY_MODEL}")
        reply = client.chat(model=PRIMARY_MODEL, messages=messages)
        
        if reply and isinstance(reply, str) and reply.strip():
            return reply.strip()
        else:
            print("⚠️ Primary model returned invalid response")
            raise Exception("Invalid response from primary model")
            
    except Exception as e:
        print(f"❌ Primary model failed: {e}")
        
        # Try fallback model (only if different from primary)
        if FALLBACK_MODEL != PRIMARY_MODEL:
            try:
                print(f"🔄 Falling back to: {FALLBACK_MODEL}")
                reply = client.chat(model=FALLBACK_MODEL, messages=messages)
                
                if reply and isinstance(reply, str) and reply.strip():
                    return reply.strip()
                else:
                    print("⚠️ Fallback model returned invalid response")
                    
            except Exception as e2:
                print(f"❌ Fallback model failed: {e2}")
        
        # Both models failed
        return "I'm having trouble connecting to my AI service right now. Please try again."

# ----------------------------
# Main Loop
# ----------------------------
def main():
    """
    Main conversation loop
    """
    print("\n" + "="*50)
    print("🤖 ShrutiBot Voice Assistant Started")
    print("="*50)
    print("Commands: Say 'quit', 'exit', or 'stop' to end")
    print("="*50 + "\n")
    
    speak("Hello! I am ShrutiBot, ready to assist you.")

    conversation_count = 0
    
    while True:
        try:
            # Listen to user
            user_text = listen_and_transcribe()
            
            if not user_text:
                speak("I didn't catch that. Please try again.")
                continue
            
            user_text_lower = user_text.lower().strip()
            
            # Check for exit commands
            if any(cmd in user_text_lower for cmd in ["quit", "exit", "stop", "goodbye", "bye"]):
                speak("Goodbye! Have a great day!")
                print("\n👋 ShrutiBot shutting down...")
                break

            # Get AI response
            conversation_count += 1
            print(f"\n--- Conversation #{conversation_count} ---")
            
            ai_response = ask_ai(user_text)
            
            # Validate and speak response
            if ai_response and ai_response.strip():
                speak(ai_response)
            else:
                fallback = "I'm not sure how to respond to that."
                speak(fallback)
            
            print()  # Add spacing between conversations

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            speak("Shutting down. Goodbye!")
            break
            
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            speak("Sorry, I encountered an error. Let's try again.")

    print("✅ ShrutiBot terminated successfully\n")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()