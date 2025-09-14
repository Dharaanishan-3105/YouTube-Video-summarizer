#!/usr/bin/env python3
"""
Deployment script for YouTube Video Summarizer
This script helps prepare the app for Streamlit Cloud deployment
"""

import os
import subprocess
import sys

def check_files():
    """Check if all required files exist."""
    required_files = [
        'youtube_summarizer_hybrid.py',
        'requirements.txt',
        'packages.txt',
        '.streamlit/config.toml',
        'setup.py',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def test_imports():
    """Test if all imports work."""
    try:
        print("Testing imports...")
        import streamlit
        import youtube_transcript_api
        import whisper
        import torch
        import transformers
        import nltk
        import spacy
        import yt_dlp
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main deployment check."""
    print("🚀 YouTube Video Summarizer - Deployment Check")
    print("=" * 50)
    
    # Check files
    if not check_files():
        print("\n❌ Deployment check failed - missing files")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Deployment check failed - import errors")
        return False
    
    print("\n✅ Deployment check passed!")
    print("\n📋 Next steps:")
    print("1. Push your code to GitHub")
    print("2. Go to https://share.streamlit.io")
    print("3. Deploy your app")
    print("4. Wait for models to download (first time only)")
    
    return True

if __name__ == "__main__":
    main()
