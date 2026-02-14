import cv2
import whisper
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from scenedetect import detect, ContentDetector
import os

class VideoProcessor:
    """
    Process lecture videos by:
    1. Extracting audio and transcribing with timestamps
    2. Detecting scene changes (slide transitions)
    3. Extracting keyframes at transitions
    4. Captioning each keyframe to understand visual content
    5. Combining into timeline-based transcript
    """
    
    def __init__(self):
        # Audio transcription
        print("Loading Whisper for transcription...")
        self.whisper_model = whisper.load_model("base")
        
        # Image captioning for frames
        print("Loading BLIP for slide understanding...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        print("Video processor ready!")
    
    def process(self, video_path):
        """
        Main processing pipeline for video lectures
        
        Returns: Dictionary with transcript, keyframes, and timeline
        """
        print(f"\\n{'='*60}")
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Step 1: Extract and transcribe audio
        print("\\n[1/4] Extracting audio and transcribing...")
        transcript_data = self._transcribe_video(video_path)
        
        # Step 2: Detect scene changes (slide transitions)
        print("\\n[2/4] Detecting scene changes (slide transitions)...")
        scene_timestamps = self._detect_scenes(video_path)
        
        # Step 3: Extract keyframes at transitions
        print("\\n[3/4] Extracting keyframes from transitions...")
        keyframes = self._extract_keyframes(video_path, scene_timestamps)
        
        # Step 4: Caption keyframes to understand slides
        print("\\n[4/4] Understanding slide content...")
        captioned_frames = self._caption_keyframes(keyframes)
        
        # Step 5: Build unified timeline
        timeline = self._build_timeline(transcript_data, captioned_frames)
        
        # Format output for knowledge base
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
                'num_slides': len(captioned_frames)
            }
        }
        
        print(f"\\n✓ Video processed successfully!")
        print(f"  - Duration: {result['duration']:.1f} seconds")
        print(f"  - Slides detected: {len(captioned_frames)}")
        print(f"  - Transcript length: {len(transcript_data['text'])} characters")
        
        return result
    
    def _transcribe_video(self, video_path):
        """
        Extract audio and transcribe with Whisper
        Returns timestamped transcript
        """
        # Transcribe directly from video (Whisper can handle video files)
        result = self.whisper_model.transcribe(
            video_path,
            language="en",
            task="transcribe",
            verbose=False
        )
        
        return {
            'text': result['text'],
            'segments': result['segments'],  # Each segment has start/end time
            'duration': result['segments'][-1]['end'] if result['segments'] else 0
        }
    
    def _detect_scenes(self, video_path, threshold=27.0):
        """
        Detect scene changes (slide transitions) in video
        
        threshold: Lower = more sensitive to changes (more scenes detected)
        27.0 is good for slide transitions
        """
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        
        # Extract timestamps (in seconds)
        timestamps = [scene[0].get_seconds() for scene in scene_list]
        
        return timestamps
    
    def _extract_keyframes(self, video_path, timestamps, max_frames=50):
        """
        Extract frames at specific timestamps
        These are typically new slides in a lecture
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        keyframes = []
        
        # Always include first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            keyframes.append({
                'timestamp': 0,
                'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            })
        
        # Extract frames at each scene change
        for timestamp in timestamps[:max_frames-1]:  # Limit to prevent too many
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                keyframes.append({
                    'timestamp': timestamp,
                    'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
                })
        
        cap.release()
        return keyframes
    
    def _caption_keyframes(self, keyframes):
        """
        Generate captions for each keyframe to understand slide content
        This tells us what's on each slide
        """
        captioned_frames = []
        
        for i, kf in enumerate(keyframes):
            # Convert numpy array to PIL Image
            image = Image.fromarray(kf['frame'])
            
            # Generate caption with BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            outputs = self.blip_model.generate(**inputs, max_new_tokens=100)
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            captioned_frames.append({
                'slide_number': i + 1,
                'timestamp': kf['timestamp'],
                'description': caption
            })
            
            print(f"  Slide {i+1} at {kf['timestamp']:.1f}s: {caption}")
        
        return captioned_frames
    
    def _build_timeline(self, transcript_data, captioned_frames):
        """
        Build unified timeline combining transcript and slides
        
        Format: [
            {time: 0, type: 'slide', content: '...'},
            {time: 2, type: 'speech', content: '...'},
            ...
        ]
        """
        timeline = []
        
        # Add slides
        for slide in captioned_frames:
            timeline.append({
                'time': slide['timestamp'],
                'type': 'slide',
                'content': f"[Slide {slide['slide_number']}] {slide['description']}"
            })
        
        # Add transcript segments
        for segment in transcript_data['segments']:
            timeline.append({
                'time': segment['start'],
                'type': 'speech',
                'content': segment['text']
            })
        
        # Sort by time
        timeline.sort(key=lambda x: x['time'])
        
        return timeline
    
    def format_for_storage(self, processed_data):
        """
        Convert processed video data into chunks for vector storage
        Each chunk represents a segment of the lecture
        """
        chunks = []
        
        # Create chunks from timeline
        for i, event in enumerate(processed_data['timeline']):
            # Create searchable text
            content = f"[{self._format_time(event['time'])}] {event['content']}"
            
            chunks.append({
                'content': content,
                'timestamp': event['time'],
                'type': event['type'],
                'source': processed_data['source'],
                'modality': 'video',
                'chunk_id': i
            })
        
        # Also create a full summary chunk
        full_transcript = processed_data['transcript']
        slides_summary = "\\n".join([
            f"Slide {s['slide_number']}: {s['description']}"
            for s in processed_data['slides']
        ])
        
        summary_content = f"""
Video Lecture: {os.path.basename(processed_data['source'])}
Duration: {processed_data['duration']:.1f} seconds

Slides Covered:
{slides_summary}

Full Transcript:
{full_transcript}
"""
        
        chunks.append({
            'content': summary_content,
            'timestamp': 0,
            'type': 'summary',
            'source': processed_data['source'],
            'modality': 'video',
            'chunk_id': 'full'
        })
        
        return chunks
    
    def _format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"