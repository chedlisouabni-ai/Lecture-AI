from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

class LectureFileHandler(FileSystemEventHandler):
    """
    Monitor folders for new lecture materials
    Automatically process when files appear
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.processing = set()
    
    def on_created(self, event):
        """Called when new file appears"""
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        # Skip temporary files
        if any(filepath.endswith(ext) for ext in ['.tmp', '.crdownload', '.part']):
            return
        
        # Skip if already processing
        if filepath in self.processing:
            return
        
        # Determine file type from folder
        if '/videos/' in filepath:
            file_type = 'video'
        elif '/audio/' in filepath:
            file_type = 'audio'
        elif '/images/' in filepath:
            file_type = 'image'
        elif '/documents/' in filepath:
            file_type = 'document'
        elif '/notes/' in filepath:
            file_type = 'text'
        else:
            print(f"Unknown file location: {filepath}")
            return
        
        print(f"\\n{'='*60}")
        print(f"NEW {file_type.upper()} FILE DETECTED!")
        print(f"File: {os.path.basename(filepath)}")
        print(f"{'='*60}")
        
        self.processing.add(filepath)
        
        # Wait for file to finish writing
        time.sleep(3)
        
        # Process the file
        try:
            self.kb.ingest_file(filepath, file_type)
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
        finally:
            self.processing.remove(filepath)

class AutoLectureWatcher:
    """
    Watch multiple folders for lecture materials
    """
    
    def __init__(self, watch_base_dir, knowledge_base):
        self.watch_base = watch_base_dir
        self.kb = knowledge_base
        self.observer = Observer()
        
        # Folders to monitor
        self.folders = ['videos', 'audio', 'images', 'documents', 'notes']
    
    def start(self):
        """Start monitoring all folders"""
        handler = LectureFileHandler(self.kb)
        
        for folder in self.folders:
            folder_path = os.path.join(self.watch_base, folder)
            os.makedirs(folder_path, exist_ok=True)
            self.observer.schedule(handler, folder_path, recursive=False)
            print(f"Watching: {folder_path}")
        
        self.observer.start()
        print("\\n✓ Auto-watcher started! Drop files to process automatically.")
    
    def stop(self):
        """Stop monitoring"""
        self.observer.stop()
        self.observer.join()