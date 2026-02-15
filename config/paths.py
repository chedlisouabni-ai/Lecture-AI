"""
Path definitions for the system
"""
from pathlib import Path
from .settings import DATA_DIR, MODELS_DIR

# Input directories
INCOMING_VIDEOS = DATA_DIR / 'incoming' / 'videos'
INCOMING_AUDIO = DATA_DIR / 'incoming' / 'audio'
INCOMING_IMAGES = DATA_DIR / 'incoming' / 'images'
INCOMING_DOCUMENTS = DATA_DIR / 'incoming' / 'documents'
INCOMING_NOTES = DATA_DIR / 'incoming' / 'notes'

# Create all incoming directories
for path in [INCOMING_VIDEOS, INCOMING_AUDIO, INCOMING_IMAGES, 
             INCOMING_DOCUMENTS, INCOMING_NOTES]:
    path.mkdir(parents=True, exist_ok=True)

# Output directories
PROCESSED_DIR = DATA_DIR / 'processed'
CACHE_DIR = DATA_DIR / 'cache'

PROCESSED_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Model directories
LLM_CACHE_DIR = MODELS_DIR / 'llm'
WHISPER_CACHE_DIR = MODELS_DIR / 'whisper'
BLIP_CACHE_DIR = MODELS_DIR / 'blip'
CLIP_CACHE_DIR = MODELS_DIR / 'clip'
EMBEDDING_CACHE_DIR = MODELS_DIR / 'embeddings'

for path in [LLM_CACHE_DIR, WHISPER_CACHE_DIR, BLIP_CACHE_DIR,
             CLIP_CACHE_DIR, EMBEDDING_CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)