# 🎓 Lecture AI - Multimodal Course Summarization System

AI MINDS Hackathon Project - Specialized lecture content processing system.

## Features
- ✅ 5 input modalities (video, audio, image, document, text)
- ✅ Automatic file monitoring and ingestion
- ✅ Local LLM processing (<4B parameters)
- ✅ Semantic search across all materials
- ✅ Natural language Q&A with sources
- ✅ Study guide generation

## Quick Start

### 1. Installation
```bash
# Clone and enter directory
cd lecture_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (system dependency)
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Download Models
```bash
python scripts/download_models.py
```
This takes 15-20 minutes. Grabs coffee ☕

### 3. Run System
```bash
python main.py
```

### 4. Add Lecture Materials
Drop files in:
- `data/incoming/videos/` - Lecture recordings
- `data/incoming/audio/` - Voice notes
- `data/incoming/images/` - Slides, diagrams
- `data/incoming/documents/` - PDFs, Word docs
- `data/incoming/notes/` - Text notes

## Usage Examples

**Ask questions:**
"What is backpropagation?"
"Summarize Lecture 3"
"What examples were given about neural networks?"

**Generate study guide:**
Click "Generate Study Guide" tab

## Technical Details

**Models Used:**
- LLM: Phi-3-mini (3.8B params) - Summarization & Q&A
- Audio: Whisper (base) - Speech-to-text
- Vision: BLIP + CLIP - Image understanding
- Embeddings: all-MiniLM-L6-v2 - Semantic search

**Storage:**
- ChromaDB for vector embeddings
- JSON for lecture metadata

## Troubleshooting

**Out of memory:**
- Reduce `WHISPER_MODEL_SIZE` to "tiny" in `.env`
- Use smaller video files
- Process fewer videos at once

**Slow processing:**
- Normal! Video processing is compute-intensive
- First-time model loading takes longer

**OCR not working:**
- Install Tesseract OCR (see installation)