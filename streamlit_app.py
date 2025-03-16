import os
import logging
import streamlit as st
import torch
import yfinance as yf
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# ✅ Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Ensure PyTorch uses CPU to prevent CUDA errors
device = "cpu"  # Force CPU mode
logger.info(f"Using device: {device}")

# ✅ Load Embedding Model (Sentence-Transformers)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ✅ Global variable to store indexed documents
document_index = []

# ✅ Step 1: Load Real Financial Data
def load_financial_data(companies: List[str]) -> List[Dict]:
    """
    Fetches financial statements from Yahoo Finance.

    Args:
        companies (List[str]): List of stock ticker symbols.

    Returns:
        List[Dict]: A list of financial document entries.
    """
    logger.info(f"Fetching financial data for companies: {companies}")

    financial_documents = []
    for ticker in companies:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract financial data (Modify as needed)
            revenue = info.get("totalRevenue", "N/A")
            profit = info.get("grossProfits", "N/A")
            summary = info.get("longBusinessSummary", "N/A")

            doc_text = f"{ticker}: Revenue = {revenue}, Profit = {profit}. {summary}"
            financial_documents.append({"company": ticker, "statement": doc_text})

            logger.info(f"✅ Data loaded for {ticker}")

        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {ticker}: {e}")

    return financial_documents

# ✅ Step 2: Index Documents with Embeddings
def index_documents():
    """
    Indexes financial documents using embeddings.
    """
    global document_index
    logger.info("Indexing financial documents...")

    # Load real data from Yahoo Finance
    companies = ["AAPL", "MSFT", "TSLA"]  # Apple, Microsoft, Tesla
    documents = load_financial_data(companies)

    # Convert text to embeddings
    for doc in documents:
        embedding = embedding_model.encode(doc["statement"], convert_to_tensor=True)
        document_index.append({"company": doc["company"], "statement": doc["statement"], "embedding": embedding})

    logger.info(f"✅ Indexed {len(document_index)} documents.")

# ✅ Step 3: Implement RAG Retrieval & Response
def retrieve_relevant_document(query: str) -> str:
    """
    Retrieves the most relevant document for the given query using embeddings.
    """
    logger.info(f"Retrieving relevant document for query: {query}")

    if not document_index:
        return "No financial data indexed yet. Try again later."

    # Convert query to embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # Find most similar document (Basic cosine similarity)
    best_match = max(document_index, key=lambda doc: torch.cosine_similarity(query_embedding, doc["embedding"], dim=0))

    logger.info(f"✅ Best match found: {best_match['statement']}")
    return best_match["statement"]

def rag_pipeline(query: str) -> str:
    """
    Generates a response based on retrieved financial statements.
    """
    relevant_doc = retrieve_relevant_document(query)
    response = f"Based on financial data: {relevant_doc}"  # TODO: Replace with AI-generated response
    return response

# ✅ Step 4: Build Streamlit UI
def main():
    try:
        # ✅ Index Documents Before Launching Streamlit
        logger.info("Starting document indexing...")
        index_documents()
        logger.info("✅ Document indexing completed successfully.")

        # ✅ Configure Streamlit UI
        st.set_page_config(page_title="Financial RAG", layout="wide")
        st.title("Financial RAG Model")
        st.write("Ask financial questions based on company statements.")

        # ✅ Setup User Input and Model Response
        user_query = st.text_input("Enter your financial question:")
        if user_query:
            logger.info(f"User Query Received: {user_query}")
            response = rag_pipeline(user_query)
            logger.info("✅ Response generated successfully.")
            st.write("### Model Response:")
            st.write(response)

        # ✅ Start Streamlit Server
        port = int(os.getenv("PORT", 8501))  # Default to 8501 if not set
        logger.info(f"Starting Streamlit app on port {port}...")
        st.write(f"🚀 App is running on port: {port}")

    except Exception as e:
        logger.error(f"❌ App failed to start: {e}", exc_info=True)
        st.error("An error occurred while starting the application. Check logs for details.")

# ✅ Run the main function
if __name__ == "__main__":
    main()
