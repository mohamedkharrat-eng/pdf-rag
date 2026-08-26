import os
import pickle
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=GROQ_API_KEY)

def load_everything():
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return embed_model, index, chunks

def retrieve(question, embed_model, index, chunks, k=TOP_K):
    query_vec = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in retrieved_chunks)
    return f"""Answer using ONLY context. If not in context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def main():
    embed_model, index, chunks = load_everything()
    print("Ready. Ask a question (or 'quit'):\n")
    
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        
        retrieved = retrieve(question, embed_model, index, chunks)
        prompt = build_prompt(question, retrieved)
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        print(f"\nGroq: {response.choices[0].message.content}\n")
        print("--- Sources ---")
        for c in retrieved:
            print(f"  {c['source']} (page {c['page']})")

if __name__ == "__main__":
    main()