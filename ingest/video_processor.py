import cv2
import os
from PIL import Image
from scenedetect import detect, ContentDetector
from typing import Dict, List
from datetime import datetime

from ingest.base_processor import BaseProcessor
from core.model_manager import ModelManager
from config.model_config import SCENE_DETECTION_CONFIG, WHISPER_CONFIG, BLIP_CONFIG

class VideoProcessor(BaseProcessor):
    """
    Process lecture videos
    Uses ModelManager for centralized model access
    """
    
    def __init__(self):
        super().__init__('video')
        
        # Get models from manager
        self.model_manager = ModelManager()
        self.whisper_model = self.model_manager.get_whisper()
        
        blip = self.model_manager.get_blip()
        self.blip_processor = blip['processor']
        self.blip_model = blip['model']
        
        self.logger.info("Video processor initialized")
    
    def process(self, video_path: str) -> Dict:
        """Process video file"""
        self.validate_file(video_path, ['mp4', 'avi', 'mov', 'mkv'])
        
        self.logger.info(f"Processing video: {os.path.basename(video_path)}")
        
        # Transcribe audio
        self.logger.info("  [1/4] Transcribing audio...")
        transcript_data = self._transcribe_video(video_path)
        
        # Detect scenes
        self.logger.info("  [2/4] Detecting scenes...")
        scene_timestamps = self._detect_scenes(video_path)
        
        # Extract keyframes
        self.logger.info("  [3/4] Extracting keyframes...")
        keyframes = self._extract_keyframes(video_path, scene_timestamps)
        
        # Caption frames
        self.logger.info("  [4/4] Captioning slides...")
        captioned_frames = self._caption_keyframes(keyframes)
        
        # Build timeline
        timeline = self._build_timeline(transcript_data, captioned_frames)
        
        result = {
            'source': video_path,
            'modality': 'video',
            'duration': transcript_data['duration'],
            'transcript': transcript_data['text'],
            'segments': transcript_data['segments'],
            'slides': captioned_frames,
            'timeline': timeline,
            'metadata': {
                'num_scenes': len(scene_timestamps),
                'num_slides': len(captioned_frames),
                'processed_at': datetime.now().isoformat()
            }
        }
        
        self.logger.info(f"✓ Video processed: {len(captioned_frames)} slides, {result['duration']:.1f}s")
        
        return result
    
    def _transcribe_video(self, video_path: str) -> Dict:
        """Transcribe video audio with Whisper"""
        result = self.whisper_model.transcribe(
            video_path,
            **WHISPER_CONFIG
        )
        
        return {
            'text': result['text'],
            'segments': result['segments'],
            'duration': result['segments'][-1]['end'] if result['segments'] else 0
        }
    
    def _detect_scenes(self, video_path: str) -> List[float]:
        """Detect scene changes"""
        scene_list = detect(
            video_path,
            ContentDetector(threshold=SCENE_DETECTION_CONFIG['threshold'])
        )
        return [scene[0].get_seconds() for scene in scene_list]
    
    def _extract_keyframes(self, video_path: str, timestamps: List[float]) -> List[Dict]:
        """Extract frames at timestamps"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []
        
        # First frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            keyframes.append({
                'timestamp': 0,
                'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            })
        
        # Scene change frames
        for timestamp in timestamps[:SCENE_DETECTION_CONFIG.get('max_scenes', 50)]:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                keyframes.append({
                    'timestamp': timestamp,
                    'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                })
        
        cap.release()
        return keyframes
    
    def _caption_keyframes(self, keyframes: List[Dict]) -> List[Dict]:
        """Generate captions for keyframes"""
        captioned_frames = []
        
        for i, kf in enumerate(keyframes):
            image = Image.fromarray(kf['frame'])
            
            inputs = self.blip_processor(image, return_tensors="pt")
            outputs = self.blip_model.generate(**inputs, max_new_tokens=BLIP_CONFIG['max_new_tokens'])
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            captioned_frames.append({
                'slide_number': i + 1,
                'timestamp': kf['timestamp'],
                'description': caption
            })
        
        return captioned_frames
    
    def _build_timeline(self, transcript_data: Dict, captioned_frames: List[Dict]) -> List[Dict]:
        """Build unified timeline"""
        timeline = []
        
        # Add slides
        for slide in captioned_frames:
            timeline.append({
                'time': slide['timestamp'],
                'type': 'slide',
                'content': f"[Slide {slide['slide_number']}] {slide['description']}"
            })
        
        # Add speech segments
        for segment in transcript_data['segments']:
            timeline.append({
                'time': segment['start'],
                'type': 'speech',
                'content': segment['text']
            })
        
        timeline.sort(key=lambda x: x['time'])
        return timeline
    
    def format_for_storage(self, processed_data: Dict) -> List[Dict]:
        """Convert to storage chunks"""
        chunks = []
        
        # Timeline chunks
        for i, event in enumerate(processed_data['timeline']):
            content = f"[{self._format_time(event['time'])}] {event['content']}"
            
            chunks.append({
                'content': content,
                'timestamp': event['time'],
                'type': event['type'],
                'source': processed_data['source'],
                'modality': 'video',
                'chunk_id': i
            })
        
        # Full summary chunk
        slides_summary = "\n".join([
            f"Slide {s['slide_number']}: {s['description']}"
            for s in processed_data['slides']
        ])
        
        summary = f"""
Video Lecture: {os.path.basename(processed_data['source'])}
Duration: {processed_data['duration']:.1f} seconds

Slides Covered:
{slides_summary}

Full Transcript:
{processed_data['transcript']}
"""
        
        chunks.append({
            'content': summary,
            'timestamp': 0,
            'type': 'summary',
            'source': processed_data['source'],
            'modality': 'video',
            'chunk_id': 'full'
        })
        
        return chunks
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"