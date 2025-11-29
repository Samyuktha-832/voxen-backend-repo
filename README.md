# 🎙️ Voxen AI - Speech-to-Speech Backend

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A FastAPI-based speech-to-speech AI backend that combines speech recognition, LLM reasoning, and text-to-speech synthesis. The system supports multiple AI providers (Ollama, OpenRouter, OpenAI) with intelligent provider switching and fallback logic.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Provider Switching](#provider-switching)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🚀 Quick Start

### Clone and Setup (One Command)

```bash
# Clone the repository
git clone https://github.com/shrutissajeev-boop/voxen_backend.git
cd voxen_backend

# One-line setup (Windows PowerShell)
py -3.11 -m venv .venv; .env\Scripts\Activate; pip install -r requirements.txt

# One-line setup (macOS/Linux Bash)
python3 -m venv .env && source .env/bin/activate && pip install -r requirements.txt
```

### Run the Server

```bash
# Activate virtual environment (if not already active)
# Windows:
.\.venv\Scripts\Activate.ps1 
# macOS/Linux:
source .env/bin/activate

# Start the server
python -m uvicorn server:app --host 127.0.0.1 --port 8000

# Server will be available at: http://127.0.0.1:8000
```

---

## 📦 Prerequisites

Before running the project, ensure you have installed:

### **1. Python (3.8 or later)**
- **Windows/macOS/Linux:** Download from [python.org](https://www.python.org/downloads/)
- **Verify installation:**
  ```bash
  python --version  # Should output Python 3.8+
  ```

### **2. Git**
- **Windows:** Download from [git-scm.com](https://git-scm.com/)
- **macOS:** `brew install git`
- **Linux:** `sudo apt install git`
- **Verify installation:**
  ```bash
  git --version
  ```

### **3. External Dependencies & Tools**

| Component | Purpose | Installation |
|-----------|---------|--------------|
| **PyAudio** | Audio input/output for speech recognition | Installed via pip (requires system audio libraries) |
| **Ollama** (optional) | Local LLM inference | Download from [ollama.ai](https://ollama.ai) |
| **FFmpeg** (optional) | Audio processing | `brew install ffmpeg` (macOS) or `choco install ffmpeg` (Windows) |

### **4. System Audio Libraries (Required for PyAudio)**

**Windows:**
```bash
# PyAudio should install automatically; if issues occur, use:
pip install pipwin
pipwin install pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-dev portaudio19-dev
```

---

## 📥 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/shrutissajeev-boop/voxen_backend.git
cd voxen_backend
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv .env
.env\Scripts\Activate
```

**macOS/Linux (Bash):**
```bash
python3 -m venv .env
source .env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

if it fails:
pip install openai-whisper pyttsx3 SpeechRecognition requests fastapi uvicorn python-multipart pydantic

pip install pyaudio
```

This installs:
- **Core AI/Speech Libraries:** Whisper, SpeechRecognition, pyttsx3, PyAudio
- **Web Framework:** FastAPI, Uvicorn
- **ML Libraries:** PyTorch, NumPy
- **Utilities:** Requests, Pydantic

### Step 4: Create Necessary Folders

```bash
# Create folders for models, data, and temporary files
mkdir -p models logs temp
```

### Step 5 (Optional): Set Up Local Ollama

If you want to use Ollama as your primary AI provider:

1. **Download and install Ollama** from [ollama.ai](https://ollama.ai)
2. **Start the Ollama service:**
   ```bash
   ollama serve
   ```
3. **In another terminal, pull a model:**
   ```bash
   ollama pull qwen2.5:0.5b  # or your preferred model
   ```

---

## 📁 Project Structure

```
voxen_backend/
├── server.py                 # Main FastAPI application
├── llm_client.py             # LLM provider abstraction layer
├── main.py                   # Speech recognition & transcription
├── config.json               # Configuration file (providers, models, settings)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env/                     # Virtual environment (created during setup)
├── models/                   # Cached AI models (Whisper, etc.)
├── logs/                     # Application logs
├── temp/                     # Temporary audio files
├── node_modules/             # JavaScript dependencies (frontend only)
└── package.json              # JavaScript dependencies
```

### Folder Purposes

| Folder | Purpose |
|--------|---------|
| `.env/` | Python virtual environment (isolated dependencies) |
| `models/` | Cached Whisper and other model files |
| `logs/` | Server and application logs |
| `temp/` | Temporary audio files during processing |
| `node_modules/` | Frontend JavaScript libraries (if applicable) |

---

## ⚙️ Configuration

### `config.json` - Provider Configuration

```json
{
  "providers": {
    "ollama": {
      "base_url": "http://localhost:11434/api",
      "default_model": "qwen2.5:0.5b",
      "num_gpu": 0,
      "num_ctx": 1024
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "sk-or-v1-YOUR-API-KEY",
      "default_model": "meta-llama/llama-3.1-70b-instruct"
    },
    "openai": {
      "api_key": "sk-YOUR-OPENAI-KEY",
      "default_model": "gpt-4o"
    }
  },
  "default_provider": "openrouter",
  "fallback_provider": "ollama",
  "force_cpu": true
}
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `default_provider` | Primary LLM provider (ollama, openrouter, openai) |
| `fallback_provider` | Backup provider if primary fails |
| `force_cpu` | Use CPU instead of GPU (set to `true` for stability) |
| `num_gpu` | Number of GPUs for Ollama (0 = CPU only) |
| `num_ctx` | Context window size for Ollama models |

### Setting API Keys

**Option 1: Update `config.json` directly (⚠️ NOT recommended for production)**
```json
"openrouter": {
  "api_key": "sk-or-v1-YOUR-ACTUAL-KEY",
  ...
}
```

**Option 2: Use environment variables (Recommended)**
```bash
# Create .env file in project root
OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY
OPENAI_API_KEY=sk-YOUR-KEY
```

**Option 3: Pass headers at runtime (Recommended for frontend)**
```bash
# Send request with x-api-key header
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-or-v1-YOUR-KEY" \
  -d '{"message":"Hello"}'
```

---

## ▶️ Running the Server

### Start the Server

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

**Expected Output:**
```
🔄 Loading Whisper model (base)...
✅ Whisper model loaded
✅ Connected to provider: ollama | model: qwen2.5:0.5b
INFO:     Started server process [1234]
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Server Configuration Options

```bash
# Run on specific port
python -m uvicorn server:app --port 9000

# Enable auto-reload during development
python -m uvicorn server:app --reload

# Allow external connections (⚠️ use with caution)
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 🔌 API Endpoints

### `POST /api/chat` - Send Message & Get Response

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is machine learning?"}'
```

**Response:**
```json
{
  "reply": "Machine learning is a branch of artificial intelligence...",
  "audio_base64": "UklGRi4A...",
  "provider_used": "ollama",
  "model_used": "qwen2.5:0.5b"
}
```

### `GET /api/listen` - Transcribe Microphone Input

**Request:**
```bash
curl http://127.0.0.1:8000/api/listen
```

**Response:**
```json
{
  "message": "What is the weather today?"
}
```

### `GET /api/profile` - Get User Profile

**Request:**
```bash
curl http://127.0.0.1:8000/api/profile
```

**Response:**
```json
{
  "user": {
    "id": "123",
    "username": "shruti",
    "full_name": "Shruti S Sajeev",
    "profile_picture": ""
  }
}
```

---

## 🔄 Provider Switching

### Switching Providers at Runtime

The system supports **intelligent provider switching** with three methods:

#### Method 1: Send `x-api-key` Header (Recommended)

```bash
# Uses OpenRouter with provided key
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "x-api-key: sk-or-v1-YOUR-KEY" \
  -d '{"message":"Hello"}'
```

#### Method 2: Use Default Provider

If no runtime key is provided, the system uses the `default_provider` from `config.json`:

```bash
# Uses default_provider (e.g., openrouter or ollama)
curl -X POST http://127.0.0.1:8000/api/chat \
  -d '{"message":"Hello"}'
```

#### Method 3: Fallback on Failure

If the primary provider fails, the system automatically tries the `fallback_provider`:

1. Primary attempt: `default_provider`
2. Fallback attempt: `fallback_provider`
3. Final fallback: Returns error message

### Flow Diagram

```
Request Received
      ↓
Is x-api-key Header Present?
      ├─ YES → Force OpenRouter with provided key
      └─ NO → Use default_provider from config.json
            ↓
      Primary Provider Succeeds?
            ├─ YES → Return Response
            └─ NO → Try fallback_provider
                  ├─ YES → Return Response
                  └─ NO → Return Error
```

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'X'"

**Cause:** Dependencies not installed or virtual environment not activated

**Fix:**
```bash
# Activate virtual environment
# Windows:
.env\Scripts\Activate
# macOS/Linux:
source .env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Issue 2: "OSError: [Errno -9997] Invalid number of channels"

**Cause:** PyAudio not properly configured or audio device not detected

**Fix:**
```bash
# Reinstall PyAudio
pip uninstall pyaudio
pip install pipwin
pipwin install pyaudio
```

**Alternative (Windows):**
```bash
pip install pipwin
pipwin install pyaudio
```

---

### Issue 3: "Connection refused" or "Failed to connect to Ollama"

**Cause:** Ollama service not running or using wrong port

**Fix:**
```bash
# 1. Check if Ollama is running
ollama serve

# 2. In another terminal, verify connection
curl http://localhost:11434/api/version

# 3. If using custom Ollama port, update config.json:
"ollama": {
  "base_url": "http://localhost:YOUR-PORT/api",
  ...
}
```

---

### Issue 4: "API key invalid" from OpenRouter

**Cause:** Invalid or expired API key

**Fix:**
```bash
# 1. Verify key format: should start with sk-or-v1-
# 2. Check API key at https://openrouter.ai/account
# 3. Test key with curl:
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer sk-or-v1-YOUR-KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/llama-3.1-70b-instruct","messages":[{"role":"user","content":"test"}]}'
```

---

### Issue 5: "Whisper model not found" or slow first load

**Cause:** Whisper model needs to be downloaded on first run

**Fix:**
```bash
# Pre-download model (runs automatically on first request)
python -c "import whisper; whisper.load_model('base')"

# Takes 1-2 minutes on first run, then cached in ./models/
```

---

### Issue 6: "Port 8000 already in use"

**Cause:** Another service is using port 8000

**Fix:**
```bash
# Use a different port
python -m uvicorn server:app --port 8001

# Or kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

---

### Issue 7: "No audio device found"

**Cause:** PyAudio can't detect microphone

**Fix:**
```bash
# 1. Check audio devices (list all)
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"

# 2. Set specific device in config or update audio capture code

# 3. On macOS, grant microphone permission:
# System Preferences → Security & Privacy → Microphone → Allow Python
```

---

### Issue 8: "torch/CUDA version mismatch"

**Cause:** PyTorch CUDA/CPU version conflicts

**Fix:**
```bash
# Install CPU-only PyTorch (simpler for most setups)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or use CUDA if you have GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Issue 9: "LLMClient failed to initialize"

**Cause:** `config.json` syntax error or provider misconfiguration

**Fix:**
```bash
# 1. Validate JSON syntax
python -c "import json; json.load(open('config.json'))"

# 2. Check config.json structure (all required fields present)

# 3. Ensure at least one provider has valid configuration

# 4. Restart server
python -m uvicorn server:app --reload
```

---

### Issue 10: Debug Mode - Enable Verbose Logging

```bash
# View detailed server logs
python -m uvicorn server:app --log-level debug

# Or set environment variable
export LOG_LEVEL=DEBUG
python -m uvicorn server:app
```

---

## 📝 Example Usage

### Python Example

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Send message and get response
def chat(message: str, api_key: str = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": message},
        headers=headers
    )
    
    data = response.json()
    print(f"Provider Used: {data['provider_used']}")
    print(f"Model Used: {data['model_used']}")
    print(f"Reply: {data['reply']}")
    return data

# Use default provider
chat("What is AI?")

# Use OpenRouter with API key
chat("Explain machine learning", api_key="sk-or-v1-YOUR-KEY")
```

### JavaScript/TypeScript Example

```javascript
const BASE_URL = "http://127.0.0.1:8000";

async function chat(message, apiKey = null) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify({ message })
  });

  const data = await response.json();
  console.log(`Provider: ${data.provider_used}, Model: ${data.model_used}`);
  console.log(`Reply: ${data.reply}`);
  return data;
}

// Usage
await chat("Hello, how are you?");
```

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/my-feature`
3. **Make your changes** and test thoroughly
4. **Commit with clear messages:** `git commit -m "Add feature: description"`
5. **Push and open a Pull Request**

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 📞 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/shrutissajeev-boop/voxen_backend/issues)
- **Email:** shrutissajeev-boop@github.com

---

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Ollama Documentation](https://ollama.ai)
- [OpenRouter API Docs](https://openrouter.ai/docs)

---

**Happy coding! 🚀**
