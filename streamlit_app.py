# Monkey-patch huggingface_hub if missing the split function
import huggingface_hub
if not hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
    def split_torch_state_dict_into_shards(state_dict, max_shard_size):
        """
        Dummy implementation for splitting a state dict into shards.
        This simply returns a list with the original state_dict.
        """
        return [state_dict]
    huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards

# Continue with the rest of your imports
import os
import logging
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Use CPU to avoid CUDA issues in Cloud environments
device = "cpu"
logger.info(f"Using device: {device}")

# Load embedding model on CPU explicitly
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Global variable for storing the indexed documents
document_index = []

# Step 1: Load Real Financial Data from Yahoo Finance
def load_financial_data(companies):
    """
    Fetch financial statements for the given tickers from Yahoo Finance.
    Returns a list of dicts with company and statement.
    """
    logger.info(f"Fetching financial data for: {companies}")
    financial_documents = []
    for ticker in companies:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Extract some financial details (customize as needed)
            revenue = info.get("totalRevenue", "N/A")
            profit = info.get("grossProfits", "N/A")
            summary = info.get("longBusinessSummary", "N/A")
            doc_text = f"{ticker}: Revenue = {revenue}, Profit = {profit}. {summary}"
            financial_documents.append({"company": ticker, "statement": doc_text})
            logger.info(f"Data loaded for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
    return financial_documents

# Step 2: Index Documents Using Embeddings
def index_documents():
    global document_index
    logger.info("Indexing financial documents...")
    companies = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    documents = load_financial_data(companies)
    for doc in documents:
        embedding = embedding_model.encode(doc["statement"], convert_to_tensor=True)
        document_index.append({"company": doc["company"], "statement": doc["statement"], "embedding": embedding})
    logger.info(f"Indexed {len(document_index)} documents.")

# Step 3: Retrieve the Most Relevant Document (Basic Cosine Similarity)
def retrieve_relevant_document(query: str) -> str:
    logger.info(f"Retrieving document for query: {query}")
    if not document_index:
        return "No financial data indexed."
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    best_match = max(document_index, key=lambda doc: torch.cosine_similarity(query_embedding, doc["embedding"], dim=0))
    logger.info(f"Best match: {best_match['statement']}")
    return best_match["statement"]

# Step 4: RAG Pipeline (Placeholder for Enhanced Generation)
def rag_pipeline(query: str) -> str:
    relevant_doc = retrieve_relevant_document(query)
    # For now, simply return the retrieved document as the response.
    response = f"Based on the financial data: {relevant_doc}"
    return response

# Step 5: Build the Streamlit UI
def main():
    try:
        logger.info("Starting document indexing...")
        index_documents()
        logger.info("Document indexing completed.")
        
        st.set_page_config(page_title="Financial RAG", layout="wide")
        st.title("Financial RAG Model")
        st.write("Ask financial questions based on real company financial statements.")
        
        user_query = st.text_input("Enter your financial question:")
        if user_query:
            logger.info(f"Received query: {user_query}")
            response = rag_pipeline(user_query)
            logger.info("Response generated.")
            st.write("### Model Response:")
            st.write(response)
            
        port = int(os.getenv("PORT", 8501))
        logger.info(f"App running on port {port}")
        st.write(f"🚀 App is running on port: {port}")
    except Exception as e:
        logger.error(f"App failed: {e}", exc_info=True)
        st.error("An error occurred. Please check logs.")

if __name__ == "__main__":
    main()