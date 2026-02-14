class TextProcessor:
    """Process plain text files (notes, transcripts)"""
    
    def process(self, text_path):
        """Read and structure text file"""
        print(f"\\nProcessing text: {os.path.basename(text_path)}")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = {
            'source': text_path,
            'modality': 'text',
            'text': text,
            'num_chars': len(text)
        }
        
        print(f"✓ Loaded {len(text)} characters")
        return result
    
    def format_for_storage(self, processed_data):
        """Chunk text for storage"""
        chunks = []
        text = processed_data['text']
        
        # Chunk by paragraphs or size
        chunk_size = 800
        overlap = 100
        
        i = 0
        chunk_id = 0
        while i < len(text):
            chunk_text = text[i:i+chunk_size]
            
            chunks.append({
                'content': chunk_text,
                'source': processed_data['source'],
                'modality': 'text',
                'chunk_id': chunk_id
            })
            
            i += chunk_size - overlap
            chunk_id += 1
        
        return chunks