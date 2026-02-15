"""
Abstract base class for all processors
Ensures consistent interface
"""
from abc import ABC, abstractmethod
from typing import Dict, List
from core.logger import logger

class BaseProcessor(ABC):
    """
    Base class for all content processors
    Enforces consistent interface across modalities
    """
    
    def __init__(self, modality: str):
        self.modality = modality
        self.logger = logger
    
    @abstractmethod
    def process(self, file_path: str) -> Dict:
        """
        Process a file and return structured data
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dict with keys: source, modality, content, metadata
        """
        pass
    
    @abstractmethod
    def format_for_storage(self, processed_data: Dict) -> List[Dict]:
        """
        Convert processed data into chunks for vector storage
        
        Args:
            processed_data: Output from process()
            
        Returns:
            List of dicts with keys: content, source, modality, chunk_id
        """
        pass
    
    def validate_file(self, file_path: str, extensions: List[str]) -> bool:
        """
        Validate file exists and has correct extension
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.split('.')[-1].lower()
        if ext not in extensions:
            raise ValueError(f"Unsupported file type: {ext}. Expected: {extensions}")
        
        return True