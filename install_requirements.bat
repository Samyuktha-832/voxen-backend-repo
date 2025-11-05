@echo off
echo ================================
echo Setting up Python environment...
echo ================================

REM Activate virtual environment
REM Make sure to activate your venv first if not already
REM Example: venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch CPU version...
python -m pip install torch==2.2.0+cpu torchvision==0.15.2+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo.
echo Installing Whisper...
python -m pip install openai-whisper

echo.
echo Installing SpeechRecognition...
python -m pip install SpeechRecognition==3.11.0

echo.
echo Installing PyAudio...
REM Change cp312 to cp313 if using Python 3.13
python -m pip install "https://github.com/intxcc/pyaudio_portaudio/releases/download/v0.2.13/PyAudio-0.2.13-cp312-cp312-win_amd64.whl"

echo.
echo Installing pyttsx3 (TTS)...
python -m pip install pyttsx3==2.90

echo.
echo Installing requests...
python -m pip install requests==2.32.0

echo.
echo Installing FastAPI, Pydantic, and Uvicorn...
python -m pip install fastapi==0.111.1 pydantic==2.3.0 uvicorn==0.26.1

echo.
echo ================================
echo All packages installed successfully!
echo ================================
echo You can now run your server with:
echo uvicorn server:app --reload
pause
