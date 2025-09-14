# 📺 YouTube Video Summarizer

A powerful Streamlit app that extracts and displays YouTube video content with timestamps, showing exactly what the author speaks/explains throughout the video.

## ✨ Features

- 🎯 **Video Content Extraction**: Get the actual transcript of what the author says
- ⏰ **Timestamps**: See when each part of the content is spoken
- 🎤 **Smart Transcript**: Uses YouTube captions first, Whisper fallback
- 🤖 **AI-Powered**: BART model for high-quality processing
- 📱 **Responsive Design**: Works on all devices
- 💾 **Export Options**: Download content as text files

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🛠️ Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd youtube-video-summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python setup.py
   ```

3. **Run the app**
   ```bash
   streamlit run youtube_summarizer_hybrid.py
   ```

## 📋 Requirements

- Python 3.8+
- Internet connection for model downloads
- ~500MB storage for AI models (one-time download)

## 🎯 How to Use

1. Enter a YouTube video URL
2. Click "Generate Video Content"
3. Get the actual video content with timestamps
4. Download as text if needed

## 🔧 Supported Video Types

- ✅ Videos with captions/subtitles (preferred)
- ✅ Videos without captions (using AI transcription)
- ✅ Long-form content (automatically processed)
- ✅ Educational content and presentations

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.