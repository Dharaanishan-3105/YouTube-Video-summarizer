#!/usr/bin/env python3
"""
YouTube Video Summarizer - Hybrid Version
- Keeps the powerful AI models for quality summaries
- Optimizes internet usage with smart caching and smaller models
- Uses efficient model loading and caching strategies
"""

import streamlit as st
import re
import os
import requests
from urllib.parse import urlparse, parse_qs
import youtube_transcript_api
from youtube_transcript_api.formatters import TextFormatter
import whisper
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import yt_dlp
import time
from datetime import timedelta
import json
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Also download the old punkt for compatibility
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer - Hybrid",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Fix white text issue and improve overall styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1000px;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 3rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: #333 !important;
    }
    
    /* Center the main content */
    .stApp > div {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .video-info {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4ecdc4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .video-info h3 {
        color: #2c3e50 !important;
        margin-bottom: 0.5rem;
    }
    
    .video-info p {
        color: #34495e !important;
        margin: 0.3rem 0;
    }
    
    .summary-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .summary-box h2 {
        color: #2c3e50 !important;
        margin-bottom: 1rem;
    }
    
    .summary-box h3 {
        color: #34495e !important;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .summary-box p {
        color: #2c3e50 !important;
        line-height: 1.6;
    }
    
    .timestamp {
        color: #4ecdc4 !important;
        font-weight: bold;
    }
    
    /* Fix main content text color */
    .main .block-container p {
        color: #2c3e50 !important;
    }
    
    .main .block-container h1, .main .block-container h2, .main .block-container h3 {
        color: #2c3e50 !important;
    }
    
    /* Improve button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
    }
    
    /* Error and warning message styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    .stAlert[data-testid="alert-error"] {
        border-left-color: #e74c3c;
        background-color: #fdf2f2;
    }
    
    .stAlert[data-testid="alert-warning"] {
        border-left-color: #f39c12;
        background-color: #fef9e7;
    }
    
    .stAlert[data-testid="alert-success"] {
        border-left-color: #27ae60;
        background-color: #f0f9f4;
    }
    
    /* Model status indicator */
    .model-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .model-loaded {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .model-loading {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class HybridYouTubeSummarizer:
    """Hybrid YouTube Summarizer with optimized internet usage and AI models."""
    
    def __init__(self):
        """Initialize the summarizer with optimized model loading."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        self.summarizer_model = None
        self.nlp = None
        self.models_initialized = False
        
        # Model configuration for internet optimization
        self.model_config = {
            'whisper_size': 'tiny',  # Use smallest Whisper model
            'summarizer_model': 'facebook/bart-large-cnn',  # Keep effective model
            'chunk_size': 800,  # Smaller chunks for efficiency
            'max_length': 150,  # Reasonable summary length
            'min_length': 30
        }
    
    def _initialize_models(self):
        """Initialize models with internet optimization."""
        if self.models_initialized:
            return
        
        try:
            # Initialize Whisper model (tiny version for speed)
            print("Loading optimized Whisper model (tiny)...")
            self.whisper_model = whisper.load_model(self.model_config['whisper_size'])
            
            # Initialize summarization model with caching
            print("Loading BART summarization model...")
            self.summarizer_model = pipeline(
                "summarization",
                model=self.model_config['summarizer_model'],
                tokenizer=self.model_config['summarizer_model'],
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
            )
            
            # Initialize spaCy for text processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy English model not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
            
            self.models_initialized = True
            print("✅ All models initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL."""
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query)['v'][0]
                elif parsed_url.path.startswith('/embed/'):
                    return parsed_url.path.split('/')[2]
            elif parsed_url.hostname == 'youtu.be':
                return parsed_url.path[1:]
        except Exception as e:
            print(f"Error extracting video ID: {e}")
        return None
    
    def extract_video_info(self, url):
        """Extract video metadata using yt-dlp."""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return None
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Format duration
                duration_seconds = info.get('duration', 0)
                duration = str(timedelta(seconds=duration_seconds)) if duration_seconds else "Unknown"
                
                # Format view count
                view_count = info.get('view_count', 0)
                views = f"{view_count:,}" if view_count else "Unknown"
                
                return {
                    'title': info.get('title', 'Unknown Title'),
                    'channel': info.get('uploader', 'Unknown Channel'),
                    'duration': duration,
                    'views': views,
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'video_id': video_id
                }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return None
    
    def extract_transcript(self, url):
        """Extract transcript with optimized fallback strategy."""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return None
            
            # Try to get transcript from YouTube first (no internet for AI)
            try:
                api = youtube_transcript_api.YouTubeTranscriptApi()
                transcript_list = api.fetch(video_id)
                formatter = TextFormatter()
                transcript = formatter.format_transcript(transcript_list)
                print("✅ Transcript extracted from YouTube captions")
                return transcript
            except Exception as e:
                print(f"YouTube transcript not available: {e}")
                print("🔄 Falling back to Whisper (tiny model)...")
                
                # Fallback to Whisper with tiny model
                return self._extract_with_whisper(url)
                
        except Exception as e:
            print(f"Error extracting transcript: {e}")
            return None
    
    def _extract_with_whisper(self, url):
        """Extract audio and transcribe using optimized Whisper."""
        try:
            if not self.models_initialized:
                self._initialize_models()
            
            # Download audio using yt-dlp with optimized settings
            ydl_opts = {
                'format': 'bestaudio[ext=mp3]/bestaudio[ext=m4a]/bestaudio',
                'outtmpl': 'temp_audio.%(ext)s',
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the downloaded audio file
            audio_file = None
            for ext in ['mp3', 'm4a', 'webm', 'wav']:
                temp_file = f"temp_audio.{ext}"
                if os.path.exists(temp_file):
                    audio_file = temp_file
                    break
            
            if not audio_file:
                print("❌ No audio file found after download")
                return None
            
            # Transcribe with tiny Whisper model
            result = self.whisper_model.transcribe(audio_file)
            
            # Clean up temporary file
            for ext in ['mp3', 'm4a', 'webm', 'wav']:
                temp_file = f"temp_audio.{ext}"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    break
            
            print("✅ Transcript extracted using Whisper (tiny)")
            return result["text"]
            
        except Exception as e:
            print(f"Error with Whisper transcription: {e}")
            return None
    
    def preprocess_transcript(self, transcript):
        """Clean and chunk the transcript for processing."""
        if not transcript:
            return []
        
        # Clean the transcript
        cleaned_text = self._clean_transcript(transcript)
        
        # Split into optimized chunks
        chunks = self._chunk_text(cleaned_text)
        
        return chunks
    
    def _clean_transcript(self, text):
        """Clean transcript by removing fillers, repetitions, and timestamps."""
        # Remove timestamps like [00:05:30]
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        
        # Remove common fillers
        fillers = ['um', 'uh', 'like', 'you know', 'so', 'well', 'actually', 'basically']
        for filler in fillers:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove repeated phrases (simple approach)
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            if sentence not in cleaned_sentences[-3:]:  # Avoid recent duplicates
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)
    
    def _chunk_text(self, text):
        """Split text into optimized chunks for the summarization model."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.model_config['chunk_size'] and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def generate_brief_summary(self, chunks, max_length=150, min_length=30):
        """Generate a brief summary using AI models."""
        if not self.models_initialized:
            self._initialize_models()
        
        if not chunks:
            return "No content available for summarization."
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                summary = self.summarizer_model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                chunk_summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                continue
        
        # Combine chunk summaries
        combined_text = ' '.join(chunk_summaries)
        
        # Create final summary
        if len(combined_text) > max_length:
            final_summary = self.summarizer_model(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return final_summary[0]['summary_text']
        else:
            return combined_text
    
    def generate_detailed_summary(self, chunks, video_info, max_length=200, min_length=50):
        """Generate a detailed summary with timestamps using AI."""
        if not self.models_initialized:
            self._initialize_models()
        
        if not chunks:
            return "No content available for summarization."
        
        # Generate summary for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = self.summarizer_model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                chunk_summaries.append({
                    'chunk_id': i,
                    'summary': summary[0]['summary_text'],
                    'content': chunk
                })
            except Exception as e:
                print(f"Error summarizing chunk {i}: {e}")
                continue
        
        # Extract key topics using spaCy
        all_text = ' '.join([chunk['content'] for chunk in chunk_summaries])
        keywords = self._extract_keywords(all_text)
        
        # Create detailed summary with timestamps
        detailed_summary = f"## 📋 Detailed Summary\n\n"
        detailed_summary += f"**🎯 Key Topics:** {', '.join(keywords[:5])}\n\n"
        
        # Add chunk summaries with estimated timestamps
        total_duration = self._parse_duration(video_info.get('duration', '0:00:00'))
        chunk_duration = total_duration / len(chunk_summaries) if chunk_summaries else 0
        
        for i, chunk_data in enumerate(chunk_summaries):
            timestamp = self._format_timestamp(i * chunk_duration)
            detailed_summary += f"### ⏰ {timestamp}\n"
            detailed_summary += f"{chunk_data['summary']}\n\n"
        
        return detailed_summary
    
    def generate_video_content_with_timestamps(self, transcript, video_info):
        """Generate video content with timestamps showing what the author speaks/explains."""
        try:
            # Clean the transcript
            cleaned_text = self._clean_transcript(transcript)
            
            # Split into sentences
            sentences = sent_tokenize(cleaned_text)
            
            # Calculate total duration
            total_duration = self._parse_duration(video_info.get('duration', '0:00:00'))
            
            # Create content with timestamps
            content = f"## 📺 Video Content: {video_info.get('title', 'Unknown Title')}\n\n"
            content += f"**Channel:** {video_info.get('channel', 'Unknown')} | **Duration:** {video_info.get('duration', 'Unknown')}\n\n"
            content += "---\n\n"
            
            # Add sentences with estimated timestamps
            if sentences:
                time_per_sentence = total_duration / len(sentences) if total_duration > 0 else 0
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():  # Skip empty sentences
                        timestamp = self._format_timestamp(i * time_per_sentence)
                        content += f"**⏰ {timestamp}**\n"
                        content += f"{sentence.strip()}\n\n"
            
            # Add key topics at the end
            if self.models_initialized:
                keywords = self._extract_keywords(cleaned_text)
                if keywords:
                    content += "---\n\n"
                    content += f"**🎯 Key Topics Mentioned:** {', '.join(keywords[:10])}\n\n"
            
            return content
            
        except Exception as e:
            print(f"Error generating video content: {e}")
            return f"Error processing video content: {str(e)}"
    
    def _extract_keywords(self, text):
        """Extract key topics using spaCy."""
        try:
            if not self.models_initialized:
                self._initialize_models()
            
            # Use spaCy for keyword extraction
            doc = self.nlp(text)
            
            # Extract nouns and proper nouns
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Count frequency and return top keywords
            keyword_counts = Counter(keywords)
            return [word for word, count in keyword_counts.most_common(10)]
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []
    
    def _parse_duration(self, duration_str):
        """Parse duration string to seconds."""
        try:
            if duration_str == "Unknown":
                return 0
            
            parts = duration_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                return int(parts[0])
        except:
            return 0
    
    def _format_timestamp(self, seconds):
        """Format seconds to MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📺 YouTube Video Summarizer</h1>', unsafe_allow_html=True)
    
    # Model status
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = HybridYouTubeSummarizer()
    
    # Center the main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Model status indicator
        if st.session_state.summarizer.models_initialized:
            st.markdown('<div class="model-status model-loaded">✅ AI Models Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="model-status model-loading">🔄 Loading Models...</div>', unsafe_allow_html=True)
            
            # Add button to preload models
            if st.button("🚀 Load AI Models Now", type="secondary", use_container_width=True):
                with st.spinner("Loading AI models... This may take a few minutes on first run."):
                    try:
                        st.session_state.summarizer._initialize_models()
                        st.success("✅ Models loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading models: {str(e)}")
                        st.info("💡 Make sure you have internet connection for first-time model download.")
        
        st.markdown("---")
        
        # Video URL input
        video_url = st.text_input(
            "🔗 Enter YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the YouTube video link you want to summarize"
        )
        
        # Process button
        process_button = st.button("🚀 Generate Video Content", type="primary", use_container_width=True)
    
    # Main content area
    if process_button and video_url:
        if not video_url.startswith(('https://www.youtube.com/', 'https://youtu.be/')):
            st.error("❌ Please enter a valid YouTube URL")
            return
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Check if models are loaded, if not, load them first
            if not st.session_state.summarizer.models_initialized:
                status_text.text("🚀 Loading AI models... (This may take a few minutes on first run)")
                progress_bar.progress(5)
                st.session_state.summarizer._initialize_models()
                status_text.text("✅ AI models loaded successfully!")
                progress_bar.progress(10)
            
            # Step 1: Extract video metadata
            status_text.text("📥 Extracting video metadata...")
            progress_bar.progress(15)
            
            video_info = st.session_state.summarizer.extract_video_info(video_url)
            if not video_info:
                st.error("❌ Could not extract video information. Please check the URL.")
                return
            
            # Display video information
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if video_info.get('thumbnail'):
                    st.image(video_info['thumbnail'], width=400)
                
                st.markdown(f"""
                <div class="video-info">
                    <h3>📺 {video_info.get('title', 'Unknown Title')}</h3>
                    <p><strong>👤 Channel:</strong> {video_info.get('channel', 'Unknown Channel')}</p>
                    <p><strong>⏱️ Duration:</strong> {video_info.get('duration', 'Unknown')}</p>
                    <p><strong>👀 Views:</strong> {video_info.get('views', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Step 2: Extract transcript
            status_text.text("📝 Extracting video content...")
            progress_bar.progress(30)
            
            transcript = st.session_state.summarizer.extract_transcript(video_url)
            if not transcript:
                st.error("❌ Could not extract video content. This could be due to:")
                st.markdown("""
                - **No captions available** for this video
                - **Audio download failed** (network issue)
                - **Video is private or restricted**
                - **Audio format not supported**
                
                **💡 Try a different video with captions or check your internet connection.**
                """)
                return
            
            # Step 3: Process transcript with timestamps
            status_text.text("🔧 Processing video content with timestamps...")
            progress_bar.progress(50)
            
            # Generate content with timestamps
            video_content = st.session_state.summarizer.generate_video_content_with_timestamps(transcript, video_info)
            
            progress_bar.progress(100)
            status_text.text("✅ Video content generated successfully!")
            
            # Display video content
            st.markdown("## 📝 Video Content with Timestamps")
            st.markdown(f'<div class="summary-box">{video_content}</div>', unsafe_allow_html=True)
            
            # Export options
            st.markdown("## 💾 Export Options")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("📄 Download as TXT", use_container_width=True):
                    txt_content = f"""
YouTube Video Content - Generated Text
=====================================

Title: {video_info.get('title', 'Unknown')}
Channel: {video_info.get('channel', 'Unknown')}
Duration: {video_info.get('duration', 'Unknown')}
URL: {video_url}

Video Content with Timestamps:
{video_content}
"""
                    st.download_button(
                        label="Download TXT",
                        data=txt_content,
                        file_name=f"youtube_content_{int(time.time())}.txt",
                        mime="text/plain"
                    )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
    
    elif process_button and not video_url:
        st.warning("⚠️ Please enter a YouTube URL first")
    
    # Simple instructions when no processing
    if not process_button:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🎯 How to Use
            
            1. **Enter YouTube URL** above
            2. **Click "Generate Video Content"**
            3. **Get the actual video content** with timestamps
            4. **Download as text** if needed
            
            **What you'll get:**
            - 📝 **Actual video content** - what the author speaks/explains
            - ⏰ **Timestamps** - when each part is said
            - 🎯 **Key topics** - main subjects discussed
            """)

if __name__ == "__main__":
    main()
