class ConceptExtractor:
    """
    Extract and organize key concepts from lectures
    Build relationships between concepts
    """
    
    def __init__(self, summarizer):
        self.summarizer = summarizer
        self.concept_graph = {}  # Store concept relationships
    
    def extract_from_lecture(self, lecture_content, lecture_id):
        """
        Extract concepts and identify relationships
        """
        # Get concepts
        concepts = self.summarizer.extract_key_concepts(lecture_content)
        
        # Store in graph
        self.concept_graph[lecture_id] = {
            'concepts': concepts,
            'content': lecture_content[:1000]  # Keep sample
        }
        
        return concepts
    
    def find_related_concepts(self, concept):
        """
        Find which lectures discuss a concept
        """
        related = []
        
        for lecture_id, data in self.concept_graph.items():
            if any(concept.lower() in c.lower() for c in data['concepts']):
                related.append({
                    'lecture': lecture_id,
                    'concepts': data['concepts']
                })
        
        return related
    
    def get_concept_progression(self, concept):
        """
        Show how a concept develops across lectures
        """
        appearances = self.find_related_concepts(concept)
        
        if not appearances:
            return f"Concept '{concept}' not found in any lectures"
        
        progression = f"Concept '{concept}' appears in:\\n"
        for app in appearances:
            progression += f"- {app['lecture']}\\n"
        
        return progression