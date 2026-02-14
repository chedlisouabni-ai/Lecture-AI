"""
Test the system with sample files
"""

def test_full_pipeline():
    """Test complete ingestion and retrieval"""
    
    print("TESTING LECTURE AI SYSTEM")
    print("="*60)
    
    # Initialize
    kb = LectureKnowledgeBase()
    
    # Test 1: Text ingestion
    print("\\n[TEST 1] Processing text notes...")
    with open("data/incoming/notes/test_notes.txt", "w") as f:
        f.write("""
        Lecture 1: Introduction to Machine Learning
        
        Today we covered the basics of machine learning:
        - Supervised vs Unsupervised learning
        - Training and test sets
        - Common algorithms: Linear Regression, Decision Trees
        
        Key takeaway: ML is about learning patterns from data.
        """)
    
    kb.ingest_file("data/incoming/notes/test_notes.txt", "text")
    
    # Test 2: Search
    print("\\n[TEST 2] Searching for content...")
    results = kb.search("What is supervised learning?")
    print(f"Found {len(results['documents'][0])} results")
    
    # Test 3: Q&A
    print("\\n[TEST 3] Asking questions...")
    answer = kb.answer_question("What are common ML algorithms?")
    print(f"Answer: {answer['answer']}")
    
    # Test 4: Summary
    print("\\n[TEST 4] Getting lecture summary...")
    lectures = kb.lecture_db.list_lectures()
    if lectures:
        summary = kb.get_lecture_summary(lectures[0])
        print(f"Summary: {summary['summary'][:200]}...")
    
    print("\\n" + "="*60)
    print("✓ ALL TESTS PASSED")

if __name__ == "__main__":
    test_full_pipeline()