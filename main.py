"""
Main entry point for Lecture AI System
Integrates all components
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.logger import logger
from core.model_manager import ModelManager
from memory.knowledge_base import LectureKnowledgeBase
from ingest.watcher import AutoLectureWatcher
from interface.app import LectureAIInterface
from config.paths import *

class LectureAISystem:
    """
    Main system orchestrator
    """
    
    def __init__(self):
        logger.info("="*60)
        logger.info("LECTURE AI SYSTEM - INITIALIZATION")
        logger.info("="*60)
        
        # Preload models
        logger.info("\n[1/4] Loading AI models...")
        try:
            model_manager = ModelManager()
            model_manager.preload_all()
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Did you run: python scripts/download_models.py ?")
            sys.exit(1)
        
        # Initialize knowledge base
        logger.info("\n[2/4] Initializing knowledge base...")
        self.knowledge_base = LectureKnowledgeBase()
        
        # Start file watcher
        logger.info("\n[3/4] Starting file monitoring...")
        self.watcher = AutoLectureWatcher(str(DATA_DIR / 'incoming'), self.knowledge_base)
        self.watcher.start()
        
        # Initialize interface
        logger.info("\n[4/4] Preparing interface...")
        self.interface = LectureAIInterface(self.knowledge_base)
        
        logger.info("\n" + "="*60)
        logger.info("✓ SYSTEM READY!")
        logger.info("="*60)
        logger.info("\nDrop files in data/incoming/ folders for automatic processing")
        logger.info("Starting web interface...\n")
    
    def run(self):
        """Start the system"""
        try:
            self.interface.launch()
        except KeyboardInterrupt:
            logger.info("\n\nShutting down...")
            self.stop()
    
    def stop(self):
        """Cleanup"""
        self.watcher.stop()
        logger.info("✓ System stopped")

def main():
    """Entry point"""
    system = LectureAISystem()
    system.run()

if __name__ == "__main__":
    main()