"""
Prompt templates for different tasks
Centralized for easy modification and consistency
"""

LECTURE_SUMMARY_PROMPT = """You are an expert at summarizing educational content. Create a comprehensive summary of this lecture.

LECTURE CONTENT:
{content}

Create a structured summary with:
1. Main Topic (1 sentence)
2. Key Concepts (bullet points)
3. Important Details
4. Definitions or formulas mentioned
5. Examples given

Summary:"""

CONCEPT_EXTRACTION_PROMPT = """Extract the key concepts, terms, and topics from this educational content.

CONTENT:
{content}

List the key concepts (one per line, numbered):
1."""

QUESTION_ANSWERING_PROMPT = """Answer this question about the lecture content using ONLY the information provided.

LECTURE CONTENT:
{context}

QUESTION: {question}

Instructions:
- Answer based only on the content above
- Be specific and reference relevant parts
- If the answer isn't in the content, say "I don't have enough information to answer this"
- Keep answer concise (2-3 sentences)
- Reference which parts of the lecture support your answer

ANSWER:"""

STUDY_GUIDE_PROMPT = """Create a comprehensive study guide from these lecture summaries.

LECTURE SUMMARIES:
{summaries}

Study guide should include:
- Overview of all topics covered
- Key concepts to remember
- Important relationships between concepts
- Suggested study focus areas

STUDY GUIDE:"""

ACTION_EXTRACTION_PROMPT = """Extract all action items, tasks, and todos from the following text:

{text}

List each action item on a new line starting with "- "
If there are no action items, respond with "No action items found."

Action items:"""

UNCERTAINTY_DETECTION_PROMPT = """Does this answer indicate uncertainty or lack of information?

ANSWER: {answer}

Respond with only: HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, or LOW_CONFIDENCE

Assessment:"""