# 👋 Chat about Mohamed Kharrat

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about me — my projects, skills, coursework, and interests — grounded in my own written profile instead of guessing.

**Live app:** [https://pdf-rag-bebeto.streamlit.app/]

## What it does

Ask questions like "What projects has Mohamed built?" or "What does Mohamed study?" and get answers pulled directly from a profile document I wrote about myself — a small, personal alternative to sending a recruiter a static resume.

## How it works

```
about_mohamed.txt → Chunk → Embed → Store in FAISS
                                          ↓
Question → Embed → Search FAISS → Retrieve top chunks
                                          ↓
                          Chunks + Question → Groq LLM → Answer
```

1. **Ingestion** (`ingest.py`) — loads the profile text file, splits it into overlapping chunks, embeds each chunk with `all-MiniLM-L6-v2` (sentence-transformers), and stores the vectors in a FAISS index.
2. **Retrieval + Generation** (`chatapp.py`) — embeds the user's question, finds the most relevant chunks via similarity search, and sends them along with the question to a Groq-hosted LLM to generate a grounded answer.

## Tech stack

- **Chunking & Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** FAISS
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **Interface:** Streamlit (chat UI, deployed on Streamlit Community Cloud)

## Project structure

```
pdf-rag/
├── ingest.py            # loads, chunks, embeds, and indexes the profile text
├── chatapp.py            # Streamlit chat interface + retrieval + generation
├── requirements.txt      # dependencies
├── about_mohamed.txt     # source profile data
├── chunks.pkl            # saved chunk text + metadata
└── vector_store.faiss    # saved FAISS vector index
```

## Running it locally

```bash
pip install -r requirements.txt
```

Add a `.env` file with your Groq API key:
```
GROQ_API_KEY=your-key-here
```

Build the index:
```bash
python ingest.py
```

Launch the chat interface:
```bash
streamlit run chatapp.py
```

## Why this project

Built to learn the full RAG pipeline end to end — chunking strategy, embeddings, vector similarity search, and LLM-grounded generation — starting from CNNs, RNNs, LSTMs, and Transformers/Attention, then applying that knowledge to a real, working system. Originally attempted on scanned ENSI course PDFs, which surfaced real data-quality challenges (OCR errors, handwriting); this version uses clean, self-written text to demonstrate the pipeline reliably.