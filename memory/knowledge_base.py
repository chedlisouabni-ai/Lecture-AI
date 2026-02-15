"""
Main knowledge base integrating all components
"""
import os
from typing import Dict, List
from datetime import datetime

from ingest import (
    VideoProcessor, 
    AudioProcessor, 
    ImageProcessor, 
    DocumentProcessor, 
    TextProcessor
)
from memory.vector_store import LectureVectorStore
from memory.lecture_database import LectureDatabase
from summarization.lecture_summarizer import LectureSummarizer
from summarization.concept_extractor import ConceptExtractor
from core.logger import logger

class LectureKnowledgeBase:
    """
    Main system integrating all processors and storage
    """
    
    def __init__(self):
        logger.info("Initializing Lecture Knowledge Base...")
        
        # Initialize all processors
        logger.info("  Loading processors...")
        self.processors = {
            'video': VideoProcessor(),
            'audio': AudioProcessor(),
            'image': ImageProcessor(),
            'document': DocumentProcessor(),
            'text': TextProcessor()
        }
        
        # Storage
        logger.info("  Connecting to vector store...")
        self.vector_store = LectureVectorStore()
        
        logger.info("  Loading lecture database...")
        self.lecture_db = LectureDatabase()
        
        # Summarization
        logger.info("  Initializing summarization engine...")
        self.summarizer = LectureSummarizer()
        self.concept_extractor = ConceptExtractor(self.summarizer)
        
        self.lecture_counter = len(self.lecture_db.list_lectures())
        
        logger.info("✓ Knowledge Base ready!")
        logger.info(f"  Existing lectures: {self.lecture_counter}")
    
    def ingest_file(self, filepath: str, file_type: str):
        """
        Process a file and add to knowledge base
        
        Args:
            filepath: Path to file
            file_type: One of: video, audio, image, document, text
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"INGESTING {file_type.upper()}: {os.path.basename(filepath)}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Process file with appropriate processor
            processor = self.processors[file_type]
            processed_data = processor.process(filepath)
            
            # Step 2: Convert to storage chunks
            chunks = processor.format_for_storage(processed_data)
            
            # Step 3: Generate lecture ID
            self.lecture_counter += 1
            lecture_id = f"Lecture_{self.lecture_counter}_{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Step 4: Store in vector database
            logger.info("\n  Storing in vector database...")
            self.vector_store.add_lecture_content(chunks, lecture_id)
            
            # Step 5: Generate summary
            logger.info("  Generating summary...")
            combined_content = "\n\n".join([c['content'] for c in chunks[:5]])  # First few chunks
            summary = self.summarizer.summarize_lecture(combined_content, lecture_id)
            
            # Step 6: Extract concepts
            logger.info("  Extracting key concepts...")
            concepts = self.concept_extractor.extract_from_lecture(combined_content, lecture_id)
            
            # Step 7: Register in lecture database
            logger.info("  Registering in lecture database...")
            self.lecture_db.add_lecture(lecture_id, {
                'title': lecture_id,
                'sources': [filepath],
                'modalities': [file_type],
                'duration': processed_data.get('duration', 0),
                'summary': summary['summary'],
                'concepts': concepts,
                'num_chunks': len(chunks)
            })
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ INGESTION COMPLETE: {lecture_id}")
            logger.info(f"{'='*60}")
            logger.info(f"Summary: {summary['summary'][:150]}...")
            logger.info(f"Key concepts: {', '.join(concepts[:5])}")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"\n❌ Error ingesting file: {e}")
            logger.exception("Full traceback:")
            raise
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search across all lectures"""
        return self.vector_store.search(query, n_results)
    
    def get_lecture_summary(self, lecture_id: str) -> Dict:
        """Get summary of a specific lecture"""
        return self.lecture_db.get_lecture(lecture_id)
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer question about course content
        
        Args:
            question: User's question
            
        Returns:
            Dict with answer, sources, confidence
        """
        logger.info(f"Answering question: {question}")
        
        # Search for relevant content
        results = self.search(question, n_results=3)
        
        if not results['documents'][0]:
            return {
                'answer': "I don't have information about that in the lectures.",
                'sources': [],
                'confidence': 'none',
                'relevant_chunks': []
            }
        
        # Combine relevant content
        context = "\n\n".join(results['documents'][0])
        
        # Generate answer
        answer = self.summarizer.answer_question(question, context)
        
        # Assess confidence
        confidence = self.summarizer.assess_confidence(answer)
        
        # Extract sources
        sources = list(set([
            meta['lecture_id'] 
            for meta in results['metadatas'][0]
        ]))
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'relevant_chunks': results['documents'][0]
        }