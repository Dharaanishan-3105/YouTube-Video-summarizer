# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account** - You need a GitHub account
2. **GitHub Repository** - Your code should be in a GitHub repository
3. **Streamlit Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** (if you haven't already)
2. **Upload your files** to the repository:
   - `youtube_summarizer_hybrid.py` (main app file)
   - `requirements.txt` (dependencies)
   - `packages.txt` (system packages)
   - `.streamlit/config.toml` (Streamlit config)
   - `setup.py` (setup script)
   - `README.md` (documentation)

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the details:**
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` or `master`
   - **Main file path**: `youtube_summarizer_hybrid.py`
   - **App URL**: Choose a custom URL (e.g., `youtube-video-summarizer`)

### Step 3: Configure Advanced Settings

1. **Click "Advanced settings"**
2. **Add environment variables** (if needed):
   - No special environment variables required for this app
3. **Set Python version**: 3.8 or higher

### Step 4: Deploy

1. **Click "Deploy!"**
2. **Wait for deployment** (5-10 minutes)
3. **Your app will be live** at `https://your-app-name.streamlit.app`

## ⚠️ Important Notes

### Model Download Time
- **First deployment** may take 10-15 minutes due to AI model downloads
- **Models are cached** after first download
- **Total download size**: ~500MB (one-time only)

### Memory Requirements
- **Streamlit Cloud** provides 1GB RAM
- **Our app uses**: ~400-600MB (fits comfortably)
- **Models loaded on-demand** to save memory

### File Size Limits
- **Streamlit Cloud limit**: 1GB total
- **Our app size**: ~50MB (well within limits)
- **Models downloaded** during runtime (not stored in repo)

## 🔧 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Check `requirements.txt` includes all dependencies
   - Ensure Python version is 3.8+

2. **"spaCy model not found"**
   - The app automatically downloads `en_core_web_sm`
   - First run may take longer

3. **"NLTK data not found"**
   - The app automatically downloads required NLTK data
   - Check internet connection

4. **Memory errors**
   - Models are loaded on-demand
   - Restart the app if needed

### Performance Tips

1. **First load** is slow (model download)
2. **Subsequent loads** are fast (models cached)
3. **Use shorter videos** for faster processing
4. **Videos with captions** process faster

## 📊 Monitoring

- **View logs** in Streamlit Cloud dashboard
- **Monitor usage** and performance
- **Check error messages** if issues occur

## 🔄 Updates

To update your deployed app:
1. **Push changes** to your GitHub repository
2. **Streamlit Cloud** automatically redeploys
3. **Wait 2-3 minutes** for new version to be live

## 🎉 Success!

Your YouTube Video Summarizer will be live at:
`https://your-app-name.streamlit.app`

**Share the link** with others to use your app!
