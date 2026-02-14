class LectureKnowledgeBase:
    """
    Main system integrating all processors and storage
    """
    
    def __init__(self):
        # Initialize all processors
        print("Initializing Lecture Knowledge Base...")
        
        self.image_processor = ImageProcessor()
        self.processors = {
            'video': VideoProcessor(),
            'audio': AudioProcessor(),
            'image': self.image_processor,
            'document': DocumentProcessor(self.image_processor),
            'text': TextProcessor()
        }
        
        # Storage
        self.vector_store = LectureVectorStore()
        self.lecture_db = LectureDatabase()
        
        # Summarization
        self.summarizer = LectureSummarizer()
        self.concept_extractor = ConceptExtractor(self.summarizer)
        
        self.lecture_counter = len(self.lecture_db.list_lectures())
        
        print("✓ Knowledge Base ready!")
    
    def ingest_file(self, filepath, file_type):
        """
        Process a file and add to knowledge base
        
        filepath: Path to file
        file_type: One of: video, audio, image, document, text
        """
        print(f"\\nIngesting {file_type}: {os.path.basename(filepath)}")
        
        # Step 1: Process file with appropriate processor
        processor = self.processors[file_type]
        processed_data = processor.process(filepath)
        
        # Step 2: Convert to storage chunks
        chunks = processor.format_for_storage(processed_data)
        
        # Step 3: Generate lecture ID
        self.lecture_counter += 1
        lecture_id = f"Lecture_{self.lecture_counter}_{file_type}"
        
        # Step 4: Store in vector database
        self.vector_store.add_lecture_content(chunks, lecture_id)
        
        # Step 5: Generate summary
        combined_content = "\\n\\n".join([c['content'] for c in chunks[:5]])  # First few chunks
        summary = self.summarizer.summarize_lecture(combined_content, lecture_id)
        
        # Step 6: Extract concepts
        concepts = self.concept_extractor.extract_from_lecture(combined_content, lecture_id)
        
        # Step 7: Register in lecture database
        self.lecture_db.add_lecture(lecture_id, {
            'title': lecture_id,
            'sources': [filepath],
            'modalities': [file_type],
            'duration': processed_data.get('duration', 0),
            'summary': summary['summary'],
            'concepts': concepts
        })
        
        print(f"\\n{'='*60}")
        print(f"✓ INGESTION COMPLETE: {lecture_id}")
        print(f"Summary: {summary['summary'][:200]}...")
        print(f"Key concepts: {', '.join(concepts[:5])}")
        print(f"{'='*60}\\n")
    
    def search(self, query, n_results=5):
        """Search across all lectures"""
        return self.vector_store.search(query, n_results)
    
    def get_lecture_summary(self, lecture_id):
        """Get summary of a specific lecture"""
        return self.lecture_db.get_lecture(lecture_id)
    
    def answer_question(self, question):
        """
        Answer question about course content
        """
        # Search for relevant content
        results = self.search(question, n_results=3)
        
        if not results['documents'][0]:
            return {
                'answer': "I don't have information about that in the lectures.",
                'sources': []
            }
        
        # Combine relevant content
        context = "\\n\\n".join(results['documents'][0])
        
        # Generate answer
        answer = self.summarizer.answer_question(question, context)
        
        # Extract sources
        sources = list(set([
            meta['lecture_id'] 
            for meta in results['metadatas'][0]
        ]))
        
        return {
            'answer': answer,
            'sources': sources,
            'relevant_chunks': results['documents'][0]
        }