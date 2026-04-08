  # 📘 Insightify

  **AI-Powered Document Intelligence & Evaluation Platform**

  <!-- [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Gemini](https://img.shields.io/badge/Gemini%202.5-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)](https://aistudio.google.com/)
  [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/) -->

  Insightify enables you to upload research documents (PDF/TXT) and interact with them intelligently. Ask contextual questions, receive logic-based challenges, and get instantly evaluated on your understanding.
</div>

---

## 📑 Table of Contents
- [About Insightify](#-about-insightify)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [API Reference](#-api-reference)
- [Prompt Engineering](#-prompt-engineering)
- [Example Use Cases](#-example-use-cases)
- [App Previews](#-app-previews)
- [License & Contact](#-license--contact)

---

## 💡 About Insightify

In today's fast-paced research environment, comprehending dense academic or technical documents is time-consuming. **Insightify** goes beyond traditional document summarization by introducing an **interactive challenge mode**. It not only provides instant summaries and answers your questions using context-grounded AI (RAG), but it also acts as an automated examiner—testing your logical reasoning and evaluating your understanding of the uploaded text.

It's the perfect companion for students, researchers, and professionals who need to critically analyze and deeply understand complex material.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **🔍 Document-Aware Understanding** | Upload `.pdf` or `.txt` documents. Insightify parses, chunks, and vectorizes the text for high-fidelity Retrieval-Augmented Generation (RAG). |
| **📝 Instant Auto-Summary** | As soon as your document is processed, receive a concise (≤150 words) top-level summary of the core concepts. |
| **❓ Ask Anything (Q&A)** | Ask free-form questions. The system retrieves relevant chunks and uses Gemini 2.5 Flash to synthesize an accurate answer. |
| **🧾 Justified Answers** | All AI responses include pinpoint references to the source material (e.g., "based on the uploaded text..."), ensuring high trust and transparency. |
| **🧠 Challenge Me Mode** | Triggers the AI to act as a university professor, generating three logic-heavy, reasoning-based questions specific to the document's nuances. |
| **🎯 Evaluation Engine** | Submit your answers to the challenge questions. Insightify compares your logic against the ideal response, providing a score (1-5) and constructive feedback. |

---

## 🏗️ System Architecture

The application is built on a robust RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Ingestion**: 
   - User uploads a `.pdf` or `.txt` file via the Streamlit frontend.
   - Text is extracted using `PyMuPDF` and split into semantically meaningful chunks using LangChain's text splitters.
2. **Embedding & Indexing**: 
   - Document chunks are converted into dense vector embeddings via the **Gemini 2.5 Embedding API**.
   - Vectors are stored and indexed locally using **FAISS** for ultra-fast similarity search.
3. **Retrieval & Q&A**: 
   - User queries are vectorized and compared against the FAISS index to retrieve the top chunks.
   - These chunks are injected into the context window of **Gemini 2.5 Flash**, prompting it to answer the user's question accurately.
4. **Challenge Generation & Evaluation**: 
   - A specialized prompt instructs the LLM to generate complex logical questions from the context.
   - Another prompt pipeline acts as an evaluator, grading the user's responses against an internally generated "ideal" answer.

---

## ⚙️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - For a rapid, interactive, and reactive web UI.
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) - High-performance asynchronous Python web framework.
- **LLM Engine**: [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) - Used for summarization, Q&A reasoning, and evaluation logic.
- **Embedding Model**: [Gemini 2.5 Embedding API](https://aistudio.google.com/) - For creating high-quality vector representations.
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) - Local, high-speed vector index.
- **Text Processing**: [LangChain](https://www.langchain.com/) (chunking logic) & [PyMuPDF](https://pymupdf.readthedocs.io/) (PDF extraction).

---

## 📁 Project Structure

```text
Insightify/
├── backend/               # FastAPI Backend Service
│   ├── main.py            # Entry point for the REST API
│   ├── routers/           # API route handlers (upload, ask, challenge)
│   └── utils/             # Helper functions (FAISS logic, PDF processing)
├── frontend/              # Streamlit Frontend UI
│   └── app.py             # Main Streamlit application
├── postman/               # Postman collection for API endpoint testing
├── preview/               # Screenshots for the README
├── test_docs/             # Sample documents for testing
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 💻 Getting Started

### Prerequisites
- Python 3.11.9+
- Git
- Google Gemini API Key. Get it for free at [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kartikeyp011/Insightify.git
   cd Insightify
   ```

2. **Set up a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your API key:
   ```env
   GEMINI_KEY=your_gemini_api_key_here
   ```

5. **Start the Backend Server (FastAPI)**
   Run this in the root directory:
   ```bash
   uvicorn backend.main:app --reload
   ```
   *The API will be available at `http://localhost:8000`*

6. **Start the Frontend UI (Streamlit)**
   Open a new terminal, ensure your virtual environment is active, and run:
   ```bash
   streamlit run frontend/app.py
   ```
   *The UI will be accessible at `http://localhost:8501`*

---

## 🔌 API Reference

Insightify comes with a decoupled backend, meaning you can integrate its document intelligence into other applications. A complete Postman API Collection is included in the repository (`postman/Insightify.postman_collection.json`).

### Main Endpoints:

- `POST /api/upload`: Uploads a `.pdf` or `.txt` file, extracts text, generates embeddings, stores them in FAISS, and returns a document summary.
- `POST /api/ask`: Accepts a user query, performs vector search against the document index, and returns a context-justified LLM response.
- `GET  /api/challenge`: Generates 3 logic-based questions based on the uploaded material context.
- `POST /api/evaluate`: Takes the generated questions and the user's answers, evaluates them, and returns a score-card with ideal answers and feedback.

---

## 🔮 Prompt Engineering

The secret sauce behind the "Challenge Me" feature lies in specialized role-playing and precise prompt engineering. 

**Generative Challenge Prompt:**
> *"You are a university professor evaluating a research paper. Based on the provided context, generate 3 logic-heavy, reasoning-based questions that test whether a student has deeply understood the arguments and implications of the text. Avoid fact-based or simple summary questions. Number them 1., 2., and 3."*

**Evaluation Prompt:**
> *"You are an examiner. Compare the following student answers with ideal responses. For each one: Return the original question; Include the user's answer; Generate the ideal answer based on the document context; Score out of 5 based on logic, relevance, and accuracy; Add a short feedback comment."*

---

## 🎯 Example Use Cases

- **Students & Academics**: Quickly ascertain the validity of a research paper and test yourself before literature review seminars.
- **Legal Professionals**: Analyze dense legal documents and interrogate the factual logic of contracts.
- **Competitive Exam Prep**: Read un-seen passages and practice logical reasoning questions instantly.
- **Curious Readers**: Dive deeper into short stories, reports, and articles.

---

## 📸 App Previews
![alt text](preview/insightify1.png)
![alt text](preview/insightify4.png)
![alt text](preview/insightify5.png)
---

## 🧑‍⚖️ License & Contact

This project is licensed under the **MIT License**.

Created by **Kartikey Narain Prajapati**

- 📧 Email: [kartikeyp011@gmail.com](mailto:kartikeyp011@gmail.com)
- 🔗 LinkedIn: [linkedin.com/in/kartikeyp011](https://www.linkedin.com/in/kartikeyp011/)
- 🔗 GitHub: [github.com/kartikeyp011](https://github.com/kartikeyp011)

<div align="center">
  <br>
  <i>If you found this tool helpful, don't forget to ⭐ star the repository!</i>
</div>