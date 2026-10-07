# resume-rag 🤖

An intelligent, interactive **RAG (Retrieval-Augmented Generation)** pipeline that lets you chat with any resume/CV using **LangChain**, **Groq LLM**, **ChromaDB**, and **HuggingFace Embeddings**.

## ✨ Features

- 📄 **PDF Parsing** — Automatically extracts text from any CV/resume PDF
- ✂️ **Smart Chunking** — Splits text intelligently to preserve context
- 🧠 **Vector Embeddings** — Uses a lightweight `all-MiniLM-L6-v2` model (~90MB) locally
- ⚡ **Groq LLM** — Ultra-fast inference via Groq's LPU hardware
- 💬 **Interactive Chat** — Ask questions about the resume in natural language
- 💾 **Persistent Vector Store** — ChromaDB stores embeddings locally

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Framework | LangChain |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| PDF Loader | PyPDFLoader |

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Ricky0731/smart_cv_qa.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Add your PDF
Place your CV/resume PDF in the project directory and update the filename in `rag.py`:
```python
pdf_file = "Your_CV.pdf"
```

### 5. Run the pipeline
```bash
python3 rag.py
```

## 🏗️ Architecture

```
PDF File
   │
   ▼
PyPDFLoader ──► Text Chunks (RecursiveCharacterTextSplitter)
                      │
                      ▼
              HuggingFace Embeddings (all-MiniLM-L6-v2)
                      │
                      ▼
                 ChromaDB (Vector Store)
                      │
            ┌─────────┘
            │   User Query ──► Retrieve Top-K Chunks
            ▼
       Groq LLM (openai/gpt-oss-120b)
            │
            ▼
        Final Answer
```

## 💡 Lessons Learned (First RAG Project!)

- Always separate **local** (embeddings) vs **API-based** (LLM) components — mixing them up will try to download 39GB of model weights to your laptop! 😅
- Use small, purpose-built embedding models — `all-MiniLM-L6-v2` is all you need for semantic search
- `os.getenv("KEY_NAME")` takes the variable **name**, not the actual key value!
- Always check the exact model namespace required by the API (e.g., `openai/gpt-oss-120b`, not just `gpt-oss-120b`)

## 📁 Project Structure

```
resume-rag/
├── rag.py              # Main RAG pipeline
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed to git!)
├── .gitignore          # Git ignore rules
├── chroma_db/          # ChromaDB vector store (auto-generated)
└── README.md           # This file
```

## 📋 Requirements

- Python 3.8+
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- ~200MB disk space for the embedding model

## 📄 License

MIT License — feel free to use, modify, and share!
