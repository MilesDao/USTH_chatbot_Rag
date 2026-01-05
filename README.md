<div align="center">
    <h1>RAG Team Intro AI – Local RAG USTH Chatbot</h1>
<hr/>
</div>

###  1. Introduction

USTH Chatbot is an AI-powered question-answering system designed for students at the University of Science and Technology of Hanoi (USTH).

The chatbot allows users to ask questions based on USTH-related documents (PDFs) such as:
 - Lecture slides
 - Course materials
 - Regulations
 - Internal documents

Key features:
 - PDF ingestion with OCR support (for scanned documents)
 - Retrieval-Augmented Generation (RAG)
 - Semantic search using vector embeddings
 - Natural language answers powered by LLMs

The system focuses on accuracy, transparency, and avoiding hallucinations by strictly grounding answers in provided documents.

### 2. Team
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Student id</th>
      <th>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Your Name</td>
      <td>Project Lead / Backend</td>
      <td>Project Lead / Backend</td>
    </tr>
    <tr>
      <td>Member 2</td>
      <td>Data Processing / OCR</td>
      <td>Data Processing / OCR</td>
    </tr>
    <tr>
      <td>Member 3</td>
      <td>Frontend / UI</td>
      <td>Frontend / UI</td>
    </tr>
    <tr>
      <td>Member 4</td>
      <td>Research / Prompt Engineering</td>
      <td>Research / Prompt Engineering</td>
    </tr>
    <tr>
      <td>Member 4</td>
      <td>Research / Prompt Engineering</td>
      <td>Research / Prompt Engineering</td>
    </tr>
    <tr>
      <td>Member 4</td>
      <td>Research / Prompt Engineering</td>
      <td>Research / Prompt Engineering</td>
    </tr>
    <tr>
      <td>Member 4</td>
      <td>Research / Prompt Engineering</td>
      <td>Research / Prompt Engineering</td>
    </tr>
  </tbody>
</table>

### 3. Installation
3.1 Requirements
 - Python ≤ 3.11 ( Python 3.12 is not supported)
 - Git
 - (Optional) NVIDIA GPU for faster embedding

3.2 Clone the repository
```bash
git clone https://github.com/your-username/usth-chatbot.git
cd usth-chatbot
```

3.3 Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3.4 Install dependencies
```bash
pip install -r requirements.txt
```
3.5 Environment variables

Create a `.env` file:
```bash 
GOOGLE_API_KEY = "Your_API_key"
```

### 4. Local Usage Example
4.1 Ingest PDF documents
Place your PDFs in the data/pdfs/ folder, then run:
```bash 
python ingestion.py
```
What happens:
 1. Load PDF files
 2. Apply OCR if needed
 3. Chunk extracted text
 4. Generate embeddings
 5. Store vectors in ChromaDB

4.2 Run the chatbot locally
```bash 
streamlit run app.py        #http://localhost:8501
```

### 5. Architecture Diagram
<p align="center">
  <img src="assets/USTH_Chatbot_RAG.drawio.png" width="400">
</p>