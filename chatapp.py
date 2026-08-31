import streamlit as st
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

GROQ_API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Chat with Mohamed", page_icon="👋")
st.title("👋 Chat about Mohamed Kharrat")
st.caption("Ask me about my projects, skills, coursework, or interests.")


@st.cache_resource
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
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks)
    return f"""You are answering questions about Mohamed Kharrat, based ONLY on the context below. Answer as if you know him well, in a friendly tone. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


embed_model, index, chunks = load_everything()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about Mohamed's projects, skills, studies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve(question, embed_model, index, chunks)
            prompt = build_prompt(question, retrieved)

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            answer = response.choices[0].message.content

            st.markdown(answer)
            with st.expander("Sources"):
                for c in retrieved:
                    st.write(f"- {c['source']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})