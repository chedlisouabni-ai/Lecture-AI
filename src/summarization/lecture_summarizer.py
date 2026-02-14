from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LectureSummarizer:
    """
    Core summarization engine for lecture content
    Uses local LLM to generate structured summaries
    """
    
    def __init__(self):
        print("Loading Phi-3-mini for summarization...")
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        print("Summarizer ready!")
    
    def generate(self, prompt, max_tokens=500):
        """Generate text from prompt"""
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def summarize_lecture(self, lecture_content, lecture_title="Lecture"):
        """
        Generate comprehensive lecture summary
        
        Input: All content from a lecture (transcript + slides + notes)
        Output: Structured summary with key points
        """
        prompt = f"""You are an expert at summarizing educational content. Create a comprehensive summary of this lecture.

LECTURE CONTENT:
{lecture_content[:3000]}  # Limit to fit in context

Create a structured summary with:
1. Main Topic (1 sentence)
2. Key Concepts (bullet points)
3. Important Details
4. Definitions or formulas mentioned
5. Examples given

Summary:"""
        
        summary = self.generate(prompt, max_tokens=600)
        
        return {
            'title': lecture_title,
            'summary': summary,
            'type': 'lecture_summary'
        }
    
    def extract_key_concepts(self, content):
        """
        Extract main concepts and terms from content
        """
        prompt = f"""Extract the key concepts, terms, and topics from this educational content.

CONTENT:
{content[:2000]}

List the key concepts (one per line):
1."""
        
        concepts = self.generate(prompt, max_tokens=200)
        
        # Parse into list
        concept_list = []
        for line in concepts.split('\\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                concept = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                if concept:
                    concept_list.append(concept)
        
        return concept_list
    
    def generate_study_guide(self, lecture_summaries):
        """
        Create a study guide from multiple lecture summaries
        """
        # Combine summaries
        combined = "\\n\\n".join([
            f"Lecture {i+1}: {s['title']}\\n{s['summary']}"
            for i, s in enumerate(lecture_summaries)
        ])
        
        prompt = f"""Create a comprehensive study guide from these lecture summaries.

LECTURE SUMMARIES:
{combined[:3500]}

Study guide should include:
- Overview of all topics covered
- Key concepts to remember
- Important relationships between concepts
- Suggested study focus areas

STUDY GUIDE:"""
        
        guide = self.generate(prompt, max_tokens=700)
        return guide
    
    def answer_question(self, question, relevant_content):
        """
        Answer a question about lecture content using retrieved context
        """
        prompt = f"""Answer this question about the lecture content using ONLY the information provided.

LECTURE CONTENT:
{relevant_content}

QUESTION: {question}

Instructions:
- Answer based only on the content above
- Be specific and reference relevant parts
- If the answer isn't in the content, say "I don't have enough information"
- Keep answer concise (2-3 sentences)

ANSWER:"""
        
        answer = self.generate(prompt, max_tokens=200)
        return answer