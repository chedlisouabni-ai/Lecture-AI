import gradio as gr

class LectureAIInterface:
    """
    User interface for lecture AI system
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def ask_question(self, question):
        """Handle Q&A"""
        result = self.kb.answer_question(question)
        
        response = f"**Answer:**\\n{result['answer']}\\n\\n"
        response += f"**Sources:**\\n"
        for source in result['sources']:
            response += f"- {source}\\n"
        
        return response
    
    def get_lecture_list(self):
        """List all lectures"""
        lectures = self.kb.lecture_db.list_lectures()
        summary = self.kb.lecture_db.get_course_summary()
        
        output = f"**Course Overview:**\\n"
        output += f"- Total lectures: {summary['total_lectures']}\\n"
        output += f"- Modalities: {', '.join(summary['modalities'])}\\n"
        output += f"- Total duration: {summary['total_duration']/60:.1f} minutes\\n\\n"
        output += "**Lectures:**\\n"
        
        for lecture_id in lectures:
            lecture = self.kb.get_lecture_summary(lecture_id)
            output += f"\\n{lecture_id}:\\n"
            output += f"  {lecture['summary'][:150]}...\\n"
        
        return output
    
    def get_lecture_summary(self, lecture_id):
        """Get detailed summary of a lecture"""
        lecture = self.kb.get_lecture_summary(lecture_id)
        
        if not lecture:
            return f"Lecture {lecture_id} not found"
        
        output = f"# {lecture['title']}\\n\\n"
        output += f"**Summary:**\\n{lecture['summary']}\\n\\n"
        output += f"**Key Concepts:**\\n"
        for concept in lecture['concepts']:
            output += f"- {concept}\\n"
        output += f"\\n**Duration:** {lecture['duration']/60:.1f} minutes\\n"
        output += f"**Sources:** {', '.join(lecture['sources'])}\\n"
        
        return output
    
    def manual_upload(self, file, file_type):
        """Manual file upload"""
        if file is None:
            return "No file selected"
        
        try:
            self.kb.ingest_file(file.name, file_type)
            return f"✓ Successfully processed: {file.name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_study_guide(self):
        """Generate comprehensive study guide"""
        lectures = self.kb.lecture_db.list_lectures()
        summaries = [self.kb.get_lecture_summary(lid) for lid in lectures]
        
        guide = self.kb.summarizer.generate_study_guide(summaries)
        return guide
    
    def launch(self):
        """Launch Gradio interface"""
        with gr.Blocks(title="Lecture AI - Course Summarization System", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🎓 Lecture AI - Intelligent Course Summarization
            Automatically process and summarize lecture materials across all formats
            """)
            
            with gr.Tab("📚 Ask Questions"):
                gr.Markdown("Ask anything about your course content")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What are the key concepts in neural networks?",
                    lines=2
                )
                question_output = gr.Markdown(label="Answer")
                ask_btn = gr.Button("Ask", variant="primary")
                ask_btn.click(self.ask_question, inputs=[question_input], outputs=[question_output])
                
                gr.Examples(
                    examples=[
                        "What is the main topic of Lecture 1?",
                        "Explain the concept of backpropagation",
                        "What examples were given about neural networks?",
                        "Summarize the key points about machine learning"
                    ],
                    inputs=[question_input]
                )
            
            with gr.Tab("📂 Upload Materials"):
                gr.Markdown("Manually upload lecture materials")
                file_input = gr.File(label="Select file")
                file_type = gr.Radio(
                    label="File type",
                    choices=["video", "audio", "image", "document", "text"],
                    value="document"
                )
                upload_output = gr.Textbox(label="Status", lines=3)
                upload_btn = gr.Button("Process File", variant="primary")
                upload_btn.click(
                    self.manual_upload,
                    inputs=[file_input, file_type],
                    outputs=[upload_output]
                )
            
            with gr.Tab("📋 Course Overview"):
                gr.Markdown("View all lectures and summaries")
                overview_output = gr.Markdown(label="Lectures")
                overview_btn = gr.Button("Refresh", variant="secondary")
                overview_btn.click(self.get_lecture_list, outputs=[overview_output])
                
                gr.Markdown("### Get Specific Lecture Summary")
                lecture_id_input = gr.Textbox(
                    label="Lecture ID",
                    placeholder="Lecture_1_video"
                )
                lecture_summary_output = gr.Markdown(label="Summary")
                summary_btn = gr.Button("Get Summary")
                summary_btn.click(
                    self.get_lecture_summary,
                    inputs=[lecture_id_input],
                    outputs=[lecture_summary_output]
                )
            
            with gr.Tab("📖 Study Guide"):
                gr.Markdown("Generate comprehensive study guide from all lectures")
                study_guide_output = gr.Markdown(label="Study Guide")
                guide_btn = gr.Button("Generate Study Guide", variant="primary")
                guide_btn.click(self.generate_study_guide, outputs=[study_guide_output])
            
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown("""
                ## How It Works
                
                ### Automatic Processing
                Drop files in these folders for automatic ingestion:
                - `data/incoming/videos/` - Lecture recordings
                - `data/incoming/audio/` - Voice recordings
                - `data/incoming/images/` - Slides, diagrams
                - `data/incoming/documents/` - PDFs, Word docs
                - `data/incoming/notes/` - Text notes
                
                ### What Gets Processed
                
                **Videos (.mp4, .avi, .mov)**
                - Audio transcription with timestamps
                - Slide detection and extraction
                - Visual content understanding
                - Timeline generation
                
                **Audio (.mp3, .wav, .m4a)**
                - Speech-to-text transcription
                - Timestamped segments
                
                **Images (.jpg, .png)**
                - Text extraction (OCR)
                - Visual description
                - Slide content analysis
                
                **Documents (.pdf, .docx)**
                - Text extraction
                - OCR for scanned pages
                - Chunked for searchability
                
                **Text (.txt, .md)**
                - Direct ingestion
                - Semantic chunking
                
                ### Features
                - Semantic search across all materials
                - Concept extraction and linking
                - Q&A with source attribution
                - Study guide generation
                - Course-wide summaries
                
                ### Tips
                - Name files descriptively (e.g., "Lecture_3_Neural_Networks.mp4")
                - Combine multiple sources (video + slides + notes) for best results
                - Ask specific questions for targeted answers
                - Use study guide feature before exams
                """)
        
        demo.launch(share=True)