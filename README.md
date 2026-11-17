
# 📚 DocMind
### LLM-Based Intelligent Question Answering System Using Context-Aware Information Retrieval

DocMind is an AI-powered document question-answering system built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF and TXT documents and ask questions based on their content.

The system retrieves relevant information from uploaded documents using FAISS vector search and generates context-aware answers using Google's Gemini LLM.

---

## ✨ Features

- **PDF & TXT Upload:** Upload and process multiple PDF and TXT documents.
- **Text Extraction:** Extract text from uploaded documents.
- **Text Chunking:** Split extracted text into smaller chunks for efficient retrieval.
- **Text Embeddings:** Generate embeddings using Google's Gemini Embedding model.
- **Vector Database:** Store and retrieve document embeddings using FAISS.
- **AI Question Answering:** Generate answers using Google Gemini.
- **Source Attribution:** Display source document names and page numbers.
- **Retrieval Strategy Comparison:** Compare Similarity Search and Maximum Marginal Relevance (MMR).
- **Answer Quality Evaluation:** Evaluate generated answers using relevance, faithfulness, reference answers, and user feedback.
- **Evaluation Report:** Export evaluation records as CSV.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Streamlit | User interface |
| LangChain | RAG pipeline |
| Google Gemini API | LLM and embeddings |
| Gemini Embedding | Text embeddings |
| FAISS | Vector database |
| PyPDF2 | PDF text extraction |
| python-dotenv | Environment variable management |

---

## 🏗️ System Architecture

```text
          User
           |
           v
    Upload PDF / TXT
           |
           v
      Text Extraction
           |
           v
       Text Chunking
           |
           v
     Gemini Embeddings
           |
           v
      FAISS Vector DB
           |
           v
      User Question
           |
           v
    Retrieval Strategy
      /           \
     v             v
Similarity        MMR
 Search           Search
      \           /
       v         v
      Relevant Context
             |
             v
       Gemini LLM
             |
             v
      Generated Answer
             |
             v
      Source Attribution
             |
             v
    Answer Evaluation
```

---

## 🔍 Retrieval Strategies

DocMind supports two retrieval strategies:

### 1. Similarity Search

Retrieves document chunks that are semantically most similar to the user's question.

### 2. Maximum Marginal Relevance (MMR)

Retrieves relevant and diverse document chunks to reduce redundant information.

Both strategies can be compared using retrieved chunks and retrieval time.

---

## 📊 Answer Quality Evaluation

DocMind provides a manual evaluation interface for assessing generated answers.

Evaluation parameters:

- **Answer Relevance:** Measures how relevant the generated answer is to the question.
- **Faithfulness:** Evaluates whether the answer is supported by the retrieved context.
- **Reference Answer:** Allows comparison against an expected answer.
- **User Feedback:** Collects helpful or not helpful feedback.

Evaluation records can be exported as a CSV report.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11
- Google Gemini API Key
- Git

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocMind
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

Add your Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from:

https://aistudio.google.com/apikey

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📖 How to Use

1. Launch the DocMind application.
2. Upload one or more PDF or TXT documents.
3. Click **Process Documents**.
4. Navigate to the Ask Documents section.
5. Enter a question related to your uploaded documents.
6. Select a retrieval strategy.
7. Generate an AI-powered answer with source information.
8. Use Retrieval Comparison to compare Similarity Search and MMR.
9. Use Answer Evaluation to assess answer quality.
10. Download evaluation records as a CSV report.

---

## 📁 Project Structure

```text
DocMind/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
└── faiss_index/       # Generated locally after processing
```

---

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| GOOGLE_API_KEY | Google Gemini API key |

Never commit your actual `.env` file or expose your API key publicly.

---

## 🚀 Future Improvements

- Persistent evaluation history.
- Automated answer quality evaluation.
- Precision@K, Recall@K and MRR retrieval metrics.
- Advanced document preprocessing.
- Support for additional document formats.
- Improved retrieval and embedding performance.
- Conversation history and multi-turn question answering.

---

## ⚠️ Limitations

- Answer quality depends on the content and quality of uploaded documents.
- Scanned PDFs may require OCR for text extraction.
- Gemini API usage is subject to rate limits and quotas.
- The current evaluation interface uses manual scoring and user feedback.
- FAISS index is stored locally and must be generated again when setting up a fresh installation.

---

## 👨‍💻 Author

**Dipangshu Das**

Computer Science & Engineering Student

---

## 📄 License

This project is developed for educational and academic purposes.
