# 📚 Chat with my ENSI PDFs

A Retrieval-Augmented Generation (RAG) system that lets you ask questions about your own course PDFs and get answers grounded in the actual course content — built from scratch to learn how RAG, embeddings, and LLMs work together.

**Live app:** https://pdf-rag-bebeto.streamlit.app/

## What it does

Upload your course PDFs once, then ask questions in plain language (French/English) and get answers pulled directly from your own notes and exams — instead of searching through hundreds of pages manually.

## How it works

```
PDFs → Chunk → Embed → Store in FAISS
                              ↓
Question → Embed → Search FAISS → Retrieve top chunks
                              ↓
                    Chunks + Question → Groq LLM → Answer
```

1. **Ingestion** (`ingest.py`) — loads PDFs, splits them into overlapping chunks, embeds each chunk with `all-MiniLM-L6-v2` (sentence-transformers), and stores the vectors in a FAISS index.
2. **Retrieval + Generation** (`chatapp.py`) — embeds the user's question, finds the most relevant chunks via similarity search, and sends them along with the question to a Groq-hosted LLM to generate a grounded answer.

## Tech stack

- **Chunking & Embeddings:** `pypdf`, `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** FAISS
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **Interface:** Streamlit (chat UI, deployed on Streamlit Community Cloud)

## Project structure

```
pdf-rag/
├── ingest.py          # loads, chunks, embeds, and indexes PDFs
├── chatapp.py          # Streamlit chat interface + retrieval + generation
├── requirements.txt    # dependencies
├── chunks.pkl           # saved chunk text + metadata
├── vector_store.faiss   # saved FAISS vector index
└── data/                # source PDFs (not tracked in git)
```

## Running it locally

```bash
pip install -r requirements.txt
```

Add a `.env` file with your Groq API key:
```
GROQ_API_KEY=your-key-here
```

Put your PDFs in `data/`, then build the index:
```bash
python ingest.py
```

Launch the chat interface:
```bash
streamlit run chatapp.py
```

## Notes

- Chunk size and overlap can be tuned in `ingest.py` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) — larger chunks help keep multi-part content (like full exam questions) intact.
- Only PDFs with an extractable text layer are usable directly; scanned/handwritten PDFs need OCR or vision-based transcription first.
- Built as a learning project to understand the full RAG pipeline: chunking strategy, embeddings, vector similarity search, and LLM-grounded generation.