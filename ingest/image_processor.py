"""
Image processor with ModelManager integration
Handles slides, whiteboard photos, diagrams
"""
import os
import torch
from PIL import Image
import pytesseract
from typing import Dict, List
from datetime import datetime

from ingest.base_processor import BaseProcessor
from core.model_manager import ModelManager
from config.model_config import BLIP_CONFIG, OCR_CONFIG
from core.logger import logger

class ImageProcessor(BaseProcessor):
    """
    Process images: slides, whiteboard photos, diagrams
    
    Two-stage processing:
    1. OCR to extract any text
    2. Image captioning to describe visual content
    """
    
    def __init__(self):
        super().__init__('image')
        
        # Get models from ModelManager
        model_manager = ModelManager()
        
        # BLIP for image captioning
        blip = model_manager.get_blip()
        self.blip_processor = blip['processor']
        self.blip_model = blip['model']
        
        # CLIP for image embeddings (optional, for future use)
        clip = model_manager.get_clip()
        self.clip_processor = clip['processor']
        self.clip_model = clip['model']
        
        logger.info("Image processor initialized")
    
    def process(self, image_path: str) -> Dict:
        """
        Extract information from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with OCR text, caption, and combined searchable text
        """
        # Validate file
        self.validate_file(image_path, ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'])
        
        logger.info(f"Processing image: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Step 1: Extract text with OCR
            logger.info("  [1/2] Extracting text with OCR...")
            ocr_text = self._extract_text_ocr(image)
            
            # Step 2: Generate visual description
            logger.info("  [2/2] Generating visual description...")
            caption = self._generate_caption(image)
            
            # Step 3: Get image embedding (for similarity search)
            embedding = self._get_image_embedding(image)
            
            # Combine information
            combined_text = self._combine_info(ocr_text, caption, image_path)
            
            result = {
                'source': image_path,
                'modality': 'image',
                'ocr_text': ocr_text,
                'caption': caption,
                'combined': combined_text,
                'embedding': embedding,
                'metadata': {
                    'has_text': len(ocr_text) > 0,
                    'ocr_length': len(ocr_text),
                    'image_size': image.size,
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✓ Image processed")
            logger.info(f"  Caption: {caption}")
            if ocr_text:
                logger.info(f"  Text found: {len(ocr_text)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def _extract_text_ocr(self, image: Image.Image) -> str:
        """
        Extract text from image using Tesseract OCR
        Useful for slides with text
        """
        try:
            text = pytesseract.image_to_string(
                image,
                lang=OCR_CONFIG['lang'],
                config=OCR_CONFIG['config']
            )
            # Clean up
            text = text.strip()
            return text if text else ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            logger.warning("Make sure Tesseract is installed: https://github.com/tesseract-ocr/tesseract")
            return ""
    
    def _generate_caption(self, image: Image.Image) -> str:
        """
        Describe what's in the image using BLIP
        Useful for diagrams, charts, photos
        """
        try:
            inputs = self.blip_processor(image, return_tensors="pt")
            outputs = self.blip_model.generate(
                **inputs, 
                max_new_tokens=BLIP_CONFIG['max_new_tokens']
            )
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"BLIP captioning failed: {e}")
            return "Image description unavailable"
    
    def _get_image_embedding(self, image: Image.Image) -> List[float]:
        """
        Get CLIP embedding for image similarity search
        """
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            # Convert to list for storage
            return image_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"CLIP embedding failed: {e}")
            return []
    
    def _combine_info(self, ocr_text: str, caption: str, source: str) -> str:
        """
        Create searchable text combining both OCR and caption
        """
        filename = os.path.basename(source)
        
        combined = f"Image: {filename}\n\n"
        combined += f"Visual Description: {caption}\n\n"
        
        if ocr_text:
            combined += f"Text Content:\n{ocr_text}\n"
        else:
            combined += "No text detected in image.\n"
        
        return combined
    
    def format_for_storage(self, processed_data: Dict) -> List[Dict]:
        """
        Convert processed image to storage format
        
        Args:
            processed_data: Output from process()
            
        Returns:
            List of chunks (typically just one for images)
        """
        return [{
            'content': processed_data['combined'],
            'source': processed_data['source'],
            'modality': 'image',
            'caption': processed_data['caption'],
            'ocr_text': processed_data['ocr_text'],
            'embedding': processed_data.get('embedding', []),
            'chunk_id': 0,
            'metadata': processed_data['metadata']
        }]