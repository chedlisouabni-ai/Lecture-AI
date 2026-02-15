"""
Test the complete system with sample files
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.knowledge_base import LectureKnowledgeBase
from core.logger import logger

def test_full_pipeline():
    """Test complete ingestion and retrieval"""
    
    logger.info("="*60)
    logger.info("TESTING LECTURE AI SYSTEM")
    logger.info("="*60)
    
    # Initialize
    kb = LectureKnowledgeBase()
    
    # Test 1: Text ingestion
    logger.info("\n[TEST 1] Processing text notes...")
    test_notes_path = "data/incoming/notes/test_notes.txt"
    
    # Create test file
    Path(test_notes_path).parent.mkdir(parents=True, exist_ok=True)
    with open(test_notes_path, "w") as f:
        f.write("""
Lecture 1: Introduction to Machine Learning

Today we covered the basics of machine learning:
- Supervised vs Unsupervised learning
- Training and test sets
- Common algorithms: Linear Regression, Decision Trees, Neural Networks

Key definitions:
- Supervised Learning: Learning from labeled data
- Unsupervised Learning: Finding patterns in unlabeled data
- Overfitting: When model performs well on training but poorly on test data

Example: We used the iris dataset to demonstrate classification.

Key takeaway: ML is about learning patterns from data to make predictions.
        """)
    
    kb.ingest_file(test_notes_path, "text")
    
    # Test 2: Search
    logger.info("\n[TEST 2] Searching for content...")
    results = kb.search("What is supervised learning?")
    logger.info(f"✓ Found {len(results['documents'][0])} results")
    
    # Test 3: Q&A
    logger.info("\n[TEST 3] Asking questions...")
    
    questions = [
        "What is supervised learning?",
        "What algorithms were mentioned?",
        "What example was given?"
    ]
    
    for question in questions:
        logger.info(f"\nQ: {question}")
        answer = kb.answer_question(question)
        logger.info(f"A: {answer['answer']}")
        logger.info(f"Confidence: {answer['confidence']}")
        logger.info(f"Sources: {answer['sources']}")
    
    # Test 4: Summary
    logger.info("\n[TEST 4] Getting lecture summary...")
    lectures = kb.lecture_db.list_lectures()
    if lectures:
        summary = kb.get_lecture_summary(lectures[0])
        logger.info(f"✓ Lecture: {summary['title']}")
        logger.info(f"  Summary: {summary['summary'][:200]}...")
        logger.info(f"  Concepts: {', '.join(summary['concepts'])}")
    
    # Test 5: Course overview
    logger.info("\n[TEST 5] Course overview...")
    course_summary = kb.lecture_db.get_course_summary()
    logger.info(f"✓ Total lectures: {course_summary['total_lectures']}")
    logger.info(f"  Modalities: {course_summary['modalities']}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)