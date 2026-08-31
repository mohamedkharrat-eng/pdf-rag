import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "."  # current folder, since data.txt is right there
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"


def load_txt_files(data_dir):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                documents.append({"text": text, "source": filename})
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
    print("Loading text files...")
    documents = load_txt_files(DATA_DIR)
    print(f"Loaded {len(documents)} text files")

    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"text": chunk, "source": doc["source"]})

    print(f"Created {len(all_chunks)} chunks total.")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print("Saved chunks.pkl")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    print(f"Saved {INDEX_PATH} — {index.ntotal} vectors")


if __name__ == "__main__":
    main()