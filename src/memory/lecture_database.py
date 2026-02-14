import json
import os
from datetime import datetime

class LectureDatabase:
    """
    Organize and track lectures
    Store metadata and relationships
    """
    
    def __init__(self, db_path="./lecture_metadata.json"):
        self.db_path = db_path
        self.lectures = self._load_db()
    
    def _load_db(self):
        """Load lecture metadata from disk"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_db(self):
        """Save metadata to disk"""
        with open(self.db_path, 'w') as f:
            json.dump(self.lectures, f, indent=2)
    
    def add_lecture(self, lecture_id, metadata):
        """
        Register a new lecture
        
        metadata should include:
        - title
        - date
        - sources (list of file paths)
        - modalities (list of types: video, audio, etc.)
        - duration (if applicable)
        """
        self.lectures[lecture_id] = {
            **metadata,
            'added_at': datetime.now().isoformat(),
            'processed': True
        }
        self._save_db()
        print(f"✓ Registered lecture: {lecture_id}")
    
    def get_lecture(self, lecture_id):
        """Get metadata for a lecture"""
        return self.lectures.get(lecture_id)
    
    def list_lectures(self):
        """List all lectures"""
        return list(self.lectures.keys())
    
    def get_course_summary(self):
        """Get overview of all lectures"""
        summary = {
            'total_lectures': len(self.lectures),
            'modalities': set(),
            'total_duration': 0
        }
        
        for lecture_data in self.lectures.values():
            summary['modalities'].update(lecture_data.get('modalities', []))
            summary['total_duration'] += lecture_data.get('duration', 0)
        
        summary['modalities'] = list(summary['modalities'])
        return summary