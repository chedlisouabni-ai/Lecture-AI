"""
Text processor with BaseProcessor inheritance
Handles plain text files and notes
"""
import os
from typing import Dict, List
from datetime import datetime

from ingest.base_processor import BaseProcessor
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from core.logger import logger

class TextProcessor(BaseProcessor):
    """
    Process plain text files (notes, transcripts, markdown)
    Simplest processor - just chunk and store
    """
    
    def __init__(self):
        super().__init__('text')
        logger.info("Text processor initialized")
    
    def process(self, text_path: str) -> Dict:
        """
        Read and structure text file
        
        Args:
            text_path: Path to text file
            
        Returns:
            Dict with text content and metadata
        """
        # Validate file
        self.validate_file(text_path, ['txt', 'md', 'markdown', 'text'])
        
        logger.info(f"Processing text: {os.path.basename(text_path)}")
        
        try:
            # Read file with UTF-8 encoding
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Count lines and words
            lines = text.split('\n')
            words = text.split()
            
            result = {
                'source': text_path,
                'modality': 'text',
                'text': text,
                'num_chars': len(text),
                'metadata': {
                    'num_lines': len(lines),
                    'num_words': len(words),
                    'encoding': 'utf-8',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✓ Loaded {len(text)} characters ({len(words)} words, {len(lines)} lines)")
            return result
            
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning("UTF-8 failed, trying latin-1 encoding...")
            with open(text_path, 'r', encoding='latin-1') as f:
                text = f.read()
            
            result = {
                'source': text_path,
                'modality': 'text',
                'text': text,
                'num_chars': len(text),
                'metadata': {
                    'encoding': 'latin-1',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✓ Loaded {len(text)} characters (latin-1 encoding)")
            return result
            
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise
    
    def format_for_storage(self, processed_data: Dict) -> List[Dict]:
        """
        Chunk text for storage
        
        Args:
            processed_data: Output from process()
            
        Returns:
            List of chunks
        """
        chunks = []
        text = processed_data['text']
        
        # Try to chunk by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if paragraphs and len(paragraphs) > 3:
            # Chunk by paragraphs
            current_chunk = ""
            chunk_id = 0
            
            for para in paragraphs:
                # If adding this paragraph would exceed chunk size, save current chunk
                if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'source': processed_data['source'],
                        'modality': 'text',
                        'chunk_id': chunk_id,
                        'metadata': {
                            'chunking_method': 'paragraph',
                            **processed_data['metadata']
                        }
                    })
                    current_chunk = para + "\n\n"
                    chunk_id += 1
                else:
                    current_chunk += para + "\n\n"
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'source': processed_data['source'],
                    'modality': 'text',
                    'chunk_id': chunk_id,
                    'metadata': {
                        'chunking_method': 'paragraph',
                        **processed_data['metadata']
                    }
                })
        else:
            # Fall back to fixed-size chunking
            chunk_id = 0
            i = 0
            
            while i < len(text):
                chunk_text = text[i:i+CHUNK_SIZE]
                
                # Try to break at sentence boundary
                if i + CHUNK_SIZE < len(text):
                    last_period = chunk_text.rfind('.')
                    last_newline = chunk_text.rfind('\n')
                    break_point = max(last_period, last_newline)
                    
                    if break_point > CHUNK_SIZE // 2:
                        chunk_text = chunk_text[:break_point+1]
                
                chunks.append({
                    'content': chunk_text.strip(),
                    'source': processed_data['source'],
                    'modality': 'text',
                    'chunk_id': chunk_id,
                    'metadata': {
                        'chunking_method': 'fixed_size',
                        'chunk_start_pos': i,
                        **processed_data['metadata']
                    }
                })
                
                i += len(chunk_text) - CHUNK_OVERLAP
                chunk_id += 1
        
        # Add full text summary
        preview = text[:500] if len(text) > 500 else text
        summary_text = f"""Text File: {os.path.basename(processed_data['source'])}
Length: {processed_data['num_chars']} characters
Words: {processed_data['metadata'].get('num_words', 'N/A')}

Preview:
{preview}...
"""
        
        chunks.append({
            'content': summary_text,
            'source': processed_data['source'],
            'modality': 'text',
            'chunk_id': 'summary',
            'metadata': processed_data['metadata']
        })
        
        logger.info(f"  Created {len(chunks)-1} content chunks + 1 summary")
        return chunks