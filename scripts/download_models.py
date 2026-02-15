"""
Download and cache all required models
Run this FIRST before anything else!
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_manager import ModelManager
from core.logger import logger
import time

def download_all_models():
    """
    Download all models required for the system
    This takes 15-20 minutes depending on internet speed
    """
    logger.info("="*60)
    logger.info("MODEL DOWNLOAD SCRIPT")
    logger.info("="*60)
    logger.info("\nThis will download approximately 8-10 GB of models.")
    logger.info("Time estimate: 15-20 minutes\n")
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    manager = ModelManager()
    
    start_time = time.time()
    
    try:
        # 1. LLM (Phi-3-mini) - ~2.5 GB
        logger.info("\n[1/5] Downloading LLM (Phi-3-mini)...")
        logger.info("Size: ~2.5 GB | ETA: 3-5 minutes")
        manager.get_llm()
        
        # 2. Whisper - ~150 MB
        logger.info("\n[2/5] Downloading Whisper (base)...")
        logger.info("Size: ~150 MB | ETA: 1 minute")
        manager.get_whisper()
        
        # 3. BLIP - ~1 GB
        logger.info("\n[3/5] Downloading BLIP...")
        logger.info("Size: ~1 GB | ETA: 2 minutes")
        manager.get_blip()
        
        # 4. CLIP - ~600 MB
        logger.info("\n[4/5] Downloading CLIP...")
        logger.info("Size: ~600 MB | ETA: 1-2 minutes")
        manager.get_clip()
        
        # 5. Embeddings - ~100 MB
        logger.info("\n[5/5] Downloading sentence transformer...")
        logger.info("Size: ~100 MB | ETA: 30 seconds")
        manager.get_embedding_model()
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info("="*60)
        logger.info("\nYou can now run: python main.py")
        
    except Exception as e:
        logger.error(f"\n❌ Error downloading models: {e}")
        logger.error("Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    download_all_models()