import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "vector_store.faiss"


def load_pdfs(data_dir):
    """Read every PDF and return one entry per page: {text, source, page}."""
    documents = []
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, filename)
        reader = PdfReader(path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({"text": text, "source": filename, "page": page_num + 1})
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2 and end < text_len:
            chunk = chunk[: last_period + 1]
            end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap if (end - overlap) > start else end

        if start >= text_len:
            break

    return [c for c in chunks if c]


def main():
    print("Loading PDFs...")
    documents = load_pdfs(DATA_DIR)
    print(f"Loaded {len(documents)} pages from {DATA_DIR}/")

    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"text": chunk, "source": doc["source"], "page": doc["page"]})

    print(f"Created {len(all_chunks)} chunks total.")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("Saved chunks.pkl")

    # --- NEW: embedding step ---
    print("\nLoading embedding model (downloads once, ~80MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding all chunks (this is the slow part, be patient)...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    print(f"Saved {INDEX_PATH} — {index.ntotal} vectors of dimension {dimension}")


if __name__ == "__main__":
    main()