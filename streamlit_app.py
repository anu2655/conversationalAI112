import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import fastapi
import streamlit as st
import re
from rank_bm25 import BM25Okapi
import numpy as np
import os

# Load Open-Source Embedding Model
EMBEDDING_MODEL = "intfloat/e5-large-v2"  # or "BAAI/bge-large-en"
embedder = SentenceTransformer(EMBEDDING_MODEL)
embedding_dim = embedder.get_sentence_embedding_dimension()

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Load Smaller Open-Source SLM (Optimized for Streamlit Cloud)
SLM_MODEL = "google/gemma-2b-it"  # Replaces Mistral-7B (too large)

# Configure 8-bit Quantization for Lower Memory Usage
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enables 8-bit mode
    bnb_4bit_compute_dtype=torch.float16
)

# Load Tokenizer and Model with Optimized Settings
tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL, token=HF_TOKEN)

try:
    model = AutoModelForCausalLM.from_pretrained(
        SLM_MODEL,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto"
    )
except RuntimeError:
    # If 8-bit fails, fallback to CPU mode
    model = AutoModelForCausalLM.from_pretrained(SLM_MODEL, token=HF_TOKEN).to("cpu")

# Sample Financial Data (Chunk Merging & Adaptive Retrieval Applied)
financial_docs = [
    "Revenue for 2023 was $5B. Q4 showed strong performance with an increase in net profit.",
    "Net income increased by 20% due to higher sales in international markets.",
    "Tesla Q4 Report states profit margins increased. Battery production scaled up significantly.",
    "Operating expenses were reduced by 10%, boosting overall earnings per share."
]

# Chunk Merging
def chunk_and_merge(documents, chunk_size=50):
    merged_chunks = []
    for doc in documents:
        words = doc.split()
        for i in range(0, len(words), chunk_size):
            merged_chunks.append(" ".join(words[i:i + chunk_size]))
    return merged_chunks

chunked_docs = chunk_and_merge(financial_docs)
tokenized_corpus = [doc.split() for doc in chunked_docs]

# BM25 for Adaptive Retrieval
bm25 = BM25Okapi(tokenized_corpus)

# Initialize FAISS Index
index = faiss.IndexFlatL2(embedding_dim)

# Index Documents
def index_documents():
    embeddings = embedder.encode(chunked_docs, convert_to_tensor=True).cpu().numpy()
    index.add(embeddings)

# Adaptive Retrieval Function
def retrieve_documents(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_embedding, top_k)
    faiss_results = [chunked_docs[i] for i in I[0]]
    bm25_results = bm25.get_top_n(query.split(), chunked_docs, n=top_k)
    combined_results = list(set(faiss_results + bm25_results))  # Merge FAISS & BM25 results
    return combined_results[:top_k]

# Guardrail: Input-Side Filtering
FORBIDDEN_TERMS = ["predict", "stock price"]
def validate_query(query):
    if any(re.search(fr"\b{term}\b", query, re.IGNORECASE) for term in FORBIDDEN_TERMS):
        return "Query contains speculative terms. Please refine your question."
    return None

# Response Generation
def generate_response(context, question):
    input_text = f"Context: {context} \n Question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# FastAPI Endpoint
app = fastapi.FastAPI()

@app.get("/query")
def query_api(q: str):
    error = validate_query(q)
    if error:
        return {"error": error}
    retrieved_docs = retrieve_documents(q)
    response = generate_response(" ".join(retrieved_docs), q)
    return {"answer": response, "sources": retrieved_docs}

# Streamlit UI
st.title("Financial RAG Q&A")
user_query = st.text_input("Ask a financial question")
if st.button("Submit"):
    error = validate_query(user_query)
    if error:
        st.error(error)
    else:
        retrieved_docs = retrieve_documents(user_query)
        response = generate_response(" ".join(retrieved_docs), user_query)
        st.write("**Answer:**", response)
        st.write("**Sources:**", retrieved_docs)

if __name__ == "__main__":
    index_documents()
