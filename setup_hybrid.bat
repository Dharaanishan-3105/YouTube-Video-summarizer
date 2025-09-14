@echo off
echo 🚀 Setting up YouTube Video Summarizer - Hybrid Version...
echo.
echo This version keeps the powerful AI models but optimizes internet usage!
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found!

REM Install dependencies
echo 📦 Installing hybrid dependencies...
pip install -r requirements_hybrid.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully!

REM Download spaCy model
echo 🌐 Downloading spaCy English model...
python -m spacy download en_core_web_sm

if errorlevel 1 (
    echo ❌ Failed to download spaCy model!
    echo Please run manually: python -m spacy download en_core_web_sm
    pause
    exit /b 1
)

echo ✅ spaCy model downloaded!

REM Download NLTK data
echo 🌐 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📖 To run the hybrid app:
echo    python -m streamlit run youtube_summarizer_hybrid.py
echo.
echo 🌐 Then open your browser to: http://localhost:8501
echo.
echo 💡 This hybrid version:
echo    - Keeps powerful AI models for quality summaries
echo    - Uses tiny Whisper model (39MB vs 139MB)
echo    - Optimizes internet usage with smart caching
echo    - Works with videos with/without captions
echo    - First run downloads models, then cached
echo.
echo 📊 Internet Usage:
echo    - First run: ~500MB (tiny Whisper + BART model)
echo    - Subsequent runs: ~1-5MB (video metadata only)
echo.
pause
