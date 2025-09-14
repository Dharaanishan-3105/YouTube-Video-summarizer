import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    """Download spaCy English model."""
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def download_nltk_data():
    """Download required NLTK data."""
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

if __name__ == "__main__":
    print("Installing requirements...")
    install_requirements()
    
    print("Downloading spaCy model...")
    download_spacy_model()
    
    print("Downloading NLTK data...")
    download_nltk_data()
    
    print("Setup complete!")
