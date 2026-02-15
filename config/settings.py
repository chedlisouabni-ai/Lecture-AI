import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face token (optional but recommended for faster downloads)
HF_TOKEN = os.getenv('HF_TOKEN', '')
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
    print("✓ HF token loaded")
else:
    print("⚠️  No HF token found. Downloads may be slower.")
    print("   Get a token at: https://huggingface.co/settings/tokens")
    print("   Add it to .env file: HF_TOKEN=your_token_here")

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', './logs'))

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
(DATA_DIR / 'incoming').mkdir(exist_ok=True)
(DATA_DIR / 'processed').mkdir(exist_ok=True)
(DATA_DIR / 'cache').mkdir(exist_ok=True)

# Model settings
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'base')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

# Processing settings
MAX_VIDEO_SCENES = int(os.getenv('MAX_VIDEO_SCENES', 50))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 800))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100))

# LLM settings
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 500))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
LLM_USE_4BIT = os.getenv('LLM_USE_4BIT', 'true').lower() == 'true'

# Database settings
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './lecture_db')
LECTURE_METADATA_PATH = os.getenv('LECTURE_METADATA_PATH', './lecture_metadata.json')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = LOGS_DIR / 'lecture_ai.log'

print(f"\nConfiguration loaded:")
print(f"  Models dir: {MODELS_DIR}")
print(f"  Data dir: {DATA_DIR}")
print(f"  LLM: {LLM_MODEL_NAME}")
print(f"  Whisper: {WHISPER_MODEL_SIZE}")
print(f"  4-bit quantization: {LLM_USE_4BIT}")