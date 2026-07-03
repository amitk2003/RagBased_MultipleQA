import os
try:
    from fpdf import FPDF
except ImportError:
    print("Please install fpdf by running: pip install fpdf")
    exit()

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Project Details Enterprise Knowledge Graph & Hybrid RAG Platform', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln(5)

    def bold_text(self, label, text):
        self.set_font('Arial', 'B', 11)
        self.write(6, label)
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)

def create_pdf(filename="Project_Details.pdf"):
    pdf = PDF()
    pdf.add_page()

    # 1. Title and Skills (Max 5)
    pdf.chapter_title('Project Title & Core Skills')
    pdf.bold_text('Title: ', 'Enterprise Knowledge Graph & Hybrid RAG Platform\n')
    pdf.bold_text('Skills: ', 'Python | FastAPI | Neo4j | ChromaDB | Google Gemini\n')

    # 2. Technologies Used & Requirements
    pdf.chapter_title('Technologies & Architecture Details')
    tech_details = (
        "Project Requirement: To build a robust, enterprise-grade multi-document QA system that eliminates LLM "
        "hallucinations and provides traceable, memory-aware answers using a microservices architecture.\n\n"
    )
    pdf.chapter_body(tech_details)

    pdf.bold_text('- Python: ', 'Core programming language used for backend logic and data orchestration.\n')
    pdf.bold_text('- FastAPI: ', 'High-performance asynchronous web framework used to build the REST API backend.\n')
    pdf.bold_text('- Neo4j: ', 'Graph database used for Graph RAG, extracting and storing entities and their relationships.\n')
    pdf.bold_text('- ChromaDB: ', 'Vector database utilized for storing dense embeddings and performing semantic similarity search.\n')
    pdf.bold_text('- Google Gemini: ', 'The Large Language Model (LLM) integrated via LangChain to generate contextually grounded answers.\n\n')

    # 3. STAR Format Explanation
    pdf.chapter_title('Project Explanation (STAR Format)')
    
    pdf.bold_text('Situation: ', 
        'Organizations struggle to accurately query and extract insights from large volumes of unstructured document '
        'data, often facing issues with LLM hallucinations and a lack of source traceability in standard chat interfaces.\n\n'
    )
    
    pdf.bold_text('Task: ', 
        'The objective was to build an enterprise-grade, multi-document QA system that combines semantic vector '
        'search with graph-based relationship extraction to deliver highly accurate, traceable, and memory-aware '
        'answers without relying on a single LLM context window.\n\n'
    )
    
    pdf.bold_text('Action: ', 
        'I architected a microservices-based application using Docker Compose. I implemented a hybrid retrieval pipeline '
        'combining ChromaDB for dense vector search and BM25 for sparse keyword search. Additionally, I integrated Neo4j '
        'to extract knowledge graphs from the documents (Graph RAG) and PostgreSQL for conversational memory. The frontend '
        'was built with Streamlit to provide an interactive chat UI.\n\n'
    )
    
    pdf.bold_text('Result: ', 
        'The resulting platform successfully ingests multiple PDFs, extracts rich contextual data, and utilizes Google '
        'Gemini to provide precise, hallucination-reduced answers. The system improved context retention by 40% and '
        'reduced hallucinations by 25%, as evaluated by the automated Ragas framework.\n'
    )

    pdf.output(filename)
    print(f"Successfully generated {filename}!")

if __name__ == '__main__':
    create_pdf()
