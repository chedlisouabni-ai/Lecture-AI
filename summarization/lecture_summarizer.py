from typing import Dict, List
import os
from summarization.llm_wrapper import LLMWrapper
from summarization.prompts import *
from core.logger import logger

class LectureSummarizer:
    """
    Main summarization engine
    Uses LLMWrapper for all text generation
    """
    
    def __init__(self):
        self.llm = LLMWrapper()
        logger.info("Lecture Summarizer initialized")
    
    def summarize_lecture(self, lecture_content: str, lecture_title: str = "Lecture") -> Dict:
        """Generate comprehensive lecture summary"""
        # Truncate content to fit in context
        max_content_length = 3000
        truncated_content = lecture_content[:max_content_length]
        if len(lecture_content) > max_content_length:
            truncated_content += "... [content truncated]"
        
        prompt = LECTURE_SUMMARY_PROMPT.format(content=truncated_content)
        summary = self.llm.generate(prompt, max_tokens=600)
        
        return {
            'title': lecture_title,
            'summary': summary,
            'type': 'lecture_summary'
        }
    
    def extract_key_concepts(self, content: str) -> List[str]:
        """Extract main concepts from content"""
        # Truncate
        content = content[:2000]
        
        prompt = CONCEPT_EXTRACTION_PROMPT.format(content=content)
        concepts_text = self.llm.generate(prompt, max_tokens=200)
        
        # Parse into list
        concept_list = []
        for line in concepts_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                concept = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                if concept:
                    concept_list.append(concept)
        
        return concept_list
    
    def answer_question(self, question: str, relevant_content: str) -> str:
        """Answer question using retrieved context"""
        # Truncate context
        max_context = 2500
        context = relevant_content[:max_context]
        
        prompt = QUESTION_ANSWERING_PROMPT.format(
            context=context,
            question=question
        )
        
        answer = self.llm.generate(prompt, max_tokens=200)
        return answer
    
    def generate_study_guide(self, lecture_summaries: List[Dict]) -> str:
        """Create study guide from multiple lectures"""
        # Combine summaries
        combined = "\n\n".join([
            f"Lecture {i+1}: {s['title']}\n{s['summary']}"
            for i, s in enumerate(lecture_summaries)
        ])
        
        # Truncate
        combined = combined[:3500]
        
        prompt = STUDY_GUIDE_PROMPT.format(summaries=combined)
        guide = self.llm.generate(prompt, max_tokens=700)
        
        return guide
    
    def assess_confidence(self, answer: str) -> str:
        """
        Assess confidence level of an answer
        Returns: 'high', 'medium', or 'low'
        """
        # Simple heuristic-based approach
        uncertainty_phrases = [
            "don't have enough information",
            "not sure",
            "cannot determine",
            "unclear",
            "don't know",
            "insufficient information"
        ]
        
        answer_lower = answer.lower()
        
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return 'low'
        
        # Could use LLM for more sophisticated assessment
        # For now, default to medium
        return 'medium'