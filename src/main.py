import os
from memory.knowledge_base import LectureKnowledgeBase
from ingest.watcher import AutoLectureWatcher
from interface.app import LectureAIInterface

class LectureAISystem:
    """
    Main system orchestrator
    """
    
    def __init__(self):
        print("="*60)
        print("INITIALIZING LECTURE AI SYSTEM")
        print("="*60)
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize knowledge base
        print("\\n[1/3] Loading AI models and knowledge base...")
        self.knowledge_base = LectureKnowledgeBase()
        
        # Initialize file watcher
        print("\\n[2/3] Starting automatic file monitoring...")
        self.watcher = AutoLectureWatcher("data/incoming", self.knowledge_base)
        self.watcher.start()
        
        # Initialize interface
        print("\\n[3/3] Preparing user interface...")
        self.interface = LectureAIInterface(self.knowledge_base)
        
        print("\\n" + "="*60)
        print("✓ SYSTEM READY!")
        print("="*60)
        print("\\nTo add lecture materials:")
        print("1. Drop files in data/incoming/ folders (automatic)")
        print("2. Use the web interface to upload manually")
        print("\\nStarting web interface...\\n")
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            "data/incoming/videos",
            "data/incoming/audio",
            "data/incoming/images",
            "data/incoming/documents",
            "data/incoming/notes",
            "data/processed"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def run(self):
        """Start the system"""
        try:
            self.interface.launch()
        except KeyboardInterrupt:
            print("\\n\\nShutting down...")
            self.stop()
    
    def stop(self):
        """Cleanup"""
        self.watcher.stop()
        print("✓ System stopped")

if __name__ == "__main__":
    system = LectureAISystem()
    system.run()