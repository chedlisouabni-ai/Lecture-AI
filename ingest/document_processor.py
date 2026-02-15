"""
Document processor with BaseProcessor inheritance
Handles PDFs and Word documents
"""
import os
from typing import Dict, List
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    import pypdf2 as PyPDF2

from docx import Document as DocxDocument
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from ingest.base_processor import BaseProcessor
from ingest.text_processor import TextProcessor
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from core.logger import logger

class DocumentProcessor(BaseProcessor):
    """
    Process PDFs and Word documents
    Handles both text-based and scanned documents
    """
    
    def __init__(self):
        super().__init__('document')
        self.text_processor = TextProcessor()
        logger.info("Document processor initialized")
    
    def process(self, doc_path: str) -> Dict:
        """
        Process document based on type
        
        Args:
            doc_path: Path to document
            
        Returns:
            Dict with extracted text and metadata
        """
        # Validate and determine type
        ext = doc_path.split('.')[-1].lower()
        self.validate_file(doc_path, ['pdf', 'docx', 'doc'])
        
        logger.info(f"Processing document: {os.path.basename(doc_path)}")
        
        if ext == 'pdf':
            return self._process_pdf(doc_path)
        elif ext in ['docx', 'doc']:
            return self._process_docx(doc_path)
        else:
            raise ValueError(f"Unsupported document type: {ext}")
    
    def _process_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF
        If text extraction fails (scanned PDF), use OCR
        """
        logger.info("  [1/2] Attempting text extraction...")
        text = ""
        num_pages = 0
        
        # Try text extraction first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {i+1} ---\n{page_text}"
                
                logger.info(f"  Extracted text from {num_pages} pages")
                
        except Exception as e:
            logger.warning(f"  Text extraction failed: {e}")
        
        # If no text extracted or very little, try OCR
        if len(text.strip()) < 100:
            logger.info("  [2/2] Falling back to OCR (scanned PDF)...")
            text = self._ocr_pdf(pdf_path)
            is_scanned = True
        else:
            is_scanned = False
        
        result = {
            'source': pdf_path,
            'modality': 'document',
            'doc_type': 'pdf',
            'text': text,
            'num_chars': len(text),
            'metadata': {
                'num_pages': num_pages,
                'is_scanned': is_scanned,
                'processed_at': datetime.now().isoformat()
            }
        }
        
        logger.info(f"✓ Extracted {len(text)} characters from PDF")
        return result
    
    def _ocr_pdf(self, pdf_path: str) -> str:
        """OCR for scanned PDFs"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text = ""
            
            for i, image in enumerate(images):
                logger.info(f"  OCR page {i+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(image)
                text += f"\n--- Page {i+1} ---\n{page_text}"
            
            return text
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            logger.error("Make sure poppler-utils is installed for pdf2image")
            return ""
    
    def _process_docx(self, docx_path: str) -> Dict:
        """Extract text from Word document"""
        logger.info("  Extracting text from Word document...")
        
        try:
            doc = DocxDocument(docx_path)
            
            # Extract paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(paragraphs)
            
            # Also extract text from tables
            table_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    table_text.append(row_text)
            
            if table_text:
                text += "\n\n--- Tables ---\n" + "\n".join(table_text)
            
            result = {
                'source': docx_path,
                'modality': 'document',
                'doc_type': 'docx',
                'text': text,
                'num_chars': len(text),
                'metadata': {
                    'num_paragraphs': len(paragraphs),
                    'num_tables': len(doc.tables),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✓ Extracted {len(text)} characters from Word document")
            return result
            
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            raise
    
    def format_for_storage(self, processed_data: Dict) -> List[Dict]:
        """
        Break document into chunks for storage
        
        Args:
            processed_data: Output from process()
            
        Returns:
            List of chunks
        """
        chunks = []
        text = processed_data['text']
        
        # Chunk by size with overlap
        chunk_id = 0
        i = 0
        
        while i < len(text):
            # Extract chunk
            chunk_text = text[i:i+CHUNK_SIZE]
            
            # Try to break at sentence boundary
            if i + CHUNK_SIZE < len(text):
                # Look for last period in chunk
                last_period = chunk_text.rfind('.')
                if last_period > CHUNK_SIZE // 2:  # Only if it's past halfway
                    chunk_text = chunk_text[:last_period+1]
            
            chunks.append({
                'content': chunk_text.strip(),
                'source': processed_data['source'],
                'modality': 'document',
                'doc_type': processed_data['doc_type'],
                'chunk_id': chunk_id,
                'metadata': {
                    'chunk_start_pos': i,
                    **processed_data['metadata']
                }
            })
            
            # Move to next chunk with overlap
            i += len(chunk_text) - CHUNK_OVERLAP
            chunk_id += 1
        
        # Add full document summary chunk
        summary_text = f"""Document: {os.path.basename(processed_data['source'])}
Type: {processed_data['doc_type'].upper()}
Pages: {processed_data['metadata'].get('num_pages', 'N/A')}
Length: {processed_data['num_chars']} characters

Content:
{text[:1000]}...
"""
        
        chunks.append({
            'content': summary_text,
            'source': processed_data['source'],
            'modality': 'document',
            'doc_type': processed_data['doc_type'],
            'chunk_id': 'summary',
            'metadata': processed_data['metadata']
        })
        
        logger.info(f"  Created {len(chunks)-1} content chunks + 1 summary")
        return chunks