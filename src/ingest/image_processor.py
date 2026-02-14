from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
import os

class ImageProcessor:
    """
    Process images: slides, whiteboard photos, diagrams
    
    Two-stage processing:
    1. OCR to extract any text
    2. Image captioning to describe visual content
    """
    
    def __init__(self):
        print("Loading BLIP for image understanding...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Image processor ready!")
    
    def process(self, image_path):
        """
        Extract information from image
        
        Supports: .jpg, .png, .jpeg, .bmp
        """
        print(f"\\nProcessing image: {os.path.basename(image_path)}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Step 1: Extract text with OCR
        print("  [1/2] Extracting text with OCR...")
        ocr_text = self._extract_text_ocr(image)
        
        # Step 2: Generate visual description
        print("  [2/2] Generating visual description...")
        caption = self._generate_caption(image)
        
        # Combine information
        combined_text = self._combine_info(ocr_text, caption, image_path)
        
        result = {
            'source': image_path,
            'modality': 'image',
            'ocr_text': ocr_text,
            'caption': caption,
            'combined': combined_text
        }
        
        print(f"✓ Image processed")
        print(f"  Caption: {caption}")
        if ocr_text:
            print(f"  Text found: {len(ocr_text)} characters")
        
        return result
    
    def _extract_text_ocr(self, image):
        """
        Extract text from image using Tesseract OCR
        Useful for slides with text
        """
        try:
            text = pytesseract.image_to_string(image)
            # Clean up
            text = text.strip()
            return text if text else ""
        except Exception as e:
            print(f"  Warning: OCR failed - {str(e)}")
            return ""
    
    def _generate_caption(self, image):
        """
        Describe what's in the image
        Useful for diagrams, charts, photos
        """
        inputs = self.blip_processor(image, return_tensors="pt")
        outputs = self.blip_model.generate(**inputs, max_new_tokens=100)
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def _combine_info(self, ocr_text, caption, source):
        """
        Create searchable text combining both OCR and caption
        """
        filename = os.path.basename(source)
        
        combined = f"Image: {filename}\\n\\n"
        combined += f"Visual Description: {caption}\\n\\n"
        
        if ocr_text:
            combined += f"Text Content:\\n{ocr_text}\\n"
        
        return combined
    
    def format_for_storage(self, processed_data):
        """Convert to storage format"""
        return [{
            'content': processed_data['combined'],
            'source': processed_data['source'],
            'modality': 'image',
            'caption': processed_data['caption'],
            'ocr_text': processed_data['ocr_text'],
            'chunk_id': 0
        }]