"""
Audio processor with ModelManager integration
Handles audio recordings and voice memos
"""
import os
from datetime import datetime
from typing import Dict, List

from ingest.base_processor import BaseProcessor
from core.model_manager import ModelManager
from config.model_config import WHISPER_CONFIG
from core.logger import logger

class AudioProcessor(BaseProcessor):
    """
    Process audio recordings (voice memos, lecture recordings)
    Uses ModelManager for Whisper model
    """
    
    def __init__(self):
        super().__init__('audio')
        
        # Get Whisper from ModelManager
        self.whisper_model = ModelManager().get_whisper()
        
        logger.info("Audio processor initialized")
    
    def process(self, audio_path: str) -> Dict:
        """
        Transcribe audio with timestamps
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcript, segments, duration
        """
        # Validate file
        self.validate_file(audio_path, ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
        
        logger.info(f"Processing audio: {os.path.basename(audio_path)}")
        
        try:
            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                audio_path,
                language=WHISPER_CONFIG['language'],
                task=WHISPER_CONFIG['task'],
                verbose=WHISPER_CONFIG['verbose'],
                word_timestamps=True  # Get word-level timing
            )
            
            # Calculate duration
            duration = result['segments'][-1]['end'] if result['segments'] else 0
            
            # Format result
            processed = {
                'source': audio_path,
                'modality': 'audio',
                'duration': duration,
                'transcript': result['text'],
                'segments': result['segments'],
                'metadata': {
                    'num_segments': len(result['segments']),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✓ Transcribed {duration:.1f}s of audio")
            logger.info(f"  Preview: {result['text'][:100]}...")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def format_for_storage(self, processed_data: Dict) -> List[Dict]:
        """
        Convert processed audio to searchable chunks
        
        Args:
            processed_data: Output from process()
            
        Returns:
            List of chunks for vector storage
        """
        chunks = []
        
        # Break transcript into segments (by sentence/pause)
        for i, segment in enumerate(processed_data['segments']):
            content = f"[{self._format_time(segment['start'])}] {segment['text']}"
            
            chunks.append({
                'content': content,
                'timestamp': segment['start'],
                'source': processed_data['source'],
                'modality': 'audio',
                'chunk_id': i,
                'metadata': {
                    'duration': segment['end'] - segment['start']
                }
            })
        
        # Add full transcript chunk for comprehensive search
        chunks.append({
            'content': f"""Audio Recording: {os.path.basename(processed_data['source'])}
Duration: {processed_data['duration']:.1f} seconds

Full Transcript:
{processed_data['transcript']}""",
            'timestamp': 0,
            'source': processed_data['source'],
            'modality': 'audio',
            'chunk_id': 'full',
            'metadata': {
                'type': 'summary',
                'num_segments': len(processed_data['segments'])
            }
        })
        
        return chunks
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"