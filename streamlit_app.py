import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st
import re
from rank_bm25 import BM25Okapi
import numpy as np
import os

# Load Open-Source Embedding Model
EMBEDDING_MODEL = "intfloat/e5-large-v2"  # or "BAAI/bge-large-en"
embedder = SentenceTransformer(EMBEDDING_MODEL)
embedding_dim = embedder.get_sentence_embedding_dimension()

# Load Hugging Face API Token from Streamlit Secrets
HF_TOKEN = st.secrets["HF_TOKEN"]  # Use secrets manager

# Load Open-Source SLM
SLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL, use_auth_token=HF_TOKEN)

# Configure 4-bit Quantization with CPU Offloading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading for large layers
)

# Load Model with Auto Device Mapping
model = AutoModelForCausalLM.from_pretrained(SLM_MODEL, use_auth_token=HF_TOKEN, device_map="auto")

# Sample Financial Data
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

# Perform indexing before running the Streamlit app
index_documents()

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
    if any(re.search(fr"\b{term}\b", query, re.IGNORECAS
