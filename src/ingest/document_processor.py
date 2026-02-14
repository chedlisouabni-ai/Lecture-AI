import pypdf2
from docx import Document
from pdf2image import convert_from_path
import pytesseract

class DocumentProcessor:
    """
    Process PDFs and Word documents
    Handles both text-based and scanned documents
    """
    
    def __init__(self, image_processor):
        self.image_processor = image_processor
    
    def process(self, doc_path):
        """Process document based on type"""
        ext = doc_path.split('.')[-1].lower()
        
        if ext == 'pdf':
            return self._process_pdf(doc_path)
        elif ext in ['docx', 'doc']:
            return self._process_docx(doc_path)
        else:
            raise ValueError(f"Unsupported document type: {ext}")
    
    def _process_pdf(self, pdf_path):
        """
        Extract text from PDF
        If text extraction fails (scanned PDF), use OCR
        """
        print(f"\\nProcessing PDF: {os.path.basename(pdf_path)}")
        
        text = ""
        
        # Try text extraction first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                print(f"  Extracting text from {num_pages} pages...")
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\\n--- Page {i+1} ---\\n{page_text}"
        except Exception as e:
            print(f"  Text extraction failed: {e}")
        
        # If no text extracted, try OCR
        if len(text.strip()) < 100:
            print("  Falling back to OCR (scanned PDF)...")
            text = self._ocr_pdf(pdf_path)
        
        result = {
            'source': pdf_path,
            'modality': 'document',
            'doc_type': 'pdf',
            'text': text,
            'num_chars': len(text)
        }
        
        print(f"✓ Extracted {len(text)} characters")
        return result
    
    def _ocr_pdf(self, pdf_path):
        """OCR for scanned PDFs"""
        images = convert_from_path(pdf_path)
        text = ""
        
        for i, image in enumerate(images):
            print(f"  OCR page {i+1}/{len(images)}...")
            page_text = pytesseract.image_to_string(image)
            text += f"\\n--- Page {i+1} ---\\n{page_text}"
        
        return text
    
    def _process_docx(self, docx_path):
        """Extract text from Word document"""
        print(f"\\nProcessing Word document: {os.path.basename(docx_path)}")
        
        doc = Document(docx_path)
        text = "\\n".join([para.text for para in doc.paragraphs])
        
        result = {
            'source': docx_path,
            'modality': 'document',
            'doc_type': 'docx',
            'text': text,
            'num_chars': len(text)
        }
        
        print(f"✓ Extracted {len(text)} characters")
        return result
    
    def format_for_storage(self, processed_data):
        """Break document into chunks for storage"""
        chunks = []
        text = processed_data['text']
        
        # Chunk by size (every 1000 characters with 200 overlap)
        chunk_size = 1000
        overlap = 200
        
        i = 0
        chunk_id = 0
        while i < len(text):
            chunk_text = text[i:i+chunk_size]
            
            chunks.append({
                'content': chunk_text,
                'source': processed_data['source'],
                'modality': 'document',
                'doc_type': processed_data['doc_type'],
                'chunk_id': chunk_id
            })
            
            i += chunk_size - overlap
            chunk_id += 1
        
        return chunks