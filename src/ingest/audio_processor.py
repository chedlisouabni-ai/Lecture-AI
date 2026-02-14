import whisper
import os
from datetime import datetime

class AudioProcessor:
    """
    Process audio recordings (voice memos, lecture recordings)
    Simpler than video - just transcription with timestamps
    """
    
    def __init__(self):
        print("Loading Whisper for audio transcription...")
        self.whisper_model = whisper.load_model("base")
        print("Audio processor ready!")
    
    def process(self, audio_path):
        """
        Transcribe audio with timestamps
        
        Supports: .mp3, .wav, .m4a, .flac, .ogg
        """
        print(f"\\nProcessing audio: {os.path.basename(audio_path)}")
        
        # Transcribe with word-level timestamps
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            verbose=False,
            word_timestamps=True  # Get word-level timing
        )
        
        # Format result
        processed = {
            'source': audio_path,
            'modality': 'audio',
            'duration': result['segments'][-1]['end'] if result['segments'] else 0,
            'transcript': result['text'],
            'segments': result['segments']
        }
        
        print(f"✓ Transcribed {processed['duration']:.1f} seconds of audio")
        print(f"  Preview: {result['text'][:100]}...")
        
        return processed
    
    def format_for_storage(self, processed_data):
        """Convert to searchable chunks"""
        chunks = []
        
        # Break transcript into segments (by sentence/pause)
        for i, segment in enumerate(processed_data['segments']):
            content = f"[{self._format_time(segment['start'])}] {segment['text']}"
            
            chunks.append({
                'content': content,
                'timestamp': segment['start'],
                'source': processed_data['source'],
                'modality': 'audio',
                'chunk_id': i
            })
        
        # Full transcript chunk
        chunks.append({
            'content': f"Audio Recording: {os.path.basename(processed_data['source'])}\\n\\nFull Transcript:\\n{processed_data['transcript']}",
            'timestamp': 0,
            'source': processed_data['source'],
            'modality': 'audio',
            'chunk_id': 'full'
        })
        
        return chunks
    
    def _format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"