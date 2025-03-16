import os
import logging
import streamlit as st
import faiss
import yfinance as yf
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ✅ Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Global variables for document storage & indexing
documents, document_embeddings, index, bm25 = [], [], None, None

def fetch_financial_data(ticker):
    """Fetch latest financial data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return f"{info['longName']} reported a revenue of {info.get('totalRevenue', 'N/A')} with net income of {info.get('netIncomeToCommon', 'N/A')}."

def index_documents():
    """Fetch financial data and index it using FAISS and BM25."""
    global documents, document_embeddings, index, bm25
    logger.info("Fetching financial data...")

    # ✅ Get real-time financial data
    companies = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    documents = [fetch_financial_data(ticker) for ticker in companies]

    # ✅ Convert to embeddings
    document_embeddings = embedding_model.encode(documents, convert_to_numpy=True)

    # ✅ Create FAISS index
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)

    # ✅ Setup BM25 for keyword-based retrieval
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    logger.info("✅ Document indexing completed successfully.")

def rag_pipeline(query):
    """Retrieve relevant document and generate a response."""
    # ✅ Convert query to embedding
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # ✅ Find the closest document using FAISS
    _, idx = index.search(query_embedding, k=1)
    retrieved_doc = documents[idx[0][0]]

    return f"Based on company financials: {retrieved_doc}"

def main():
    try:
        # ✅ Step 1: Index Documents Before Launching Streamlit
        logger.info("Starting document indexing...")
        index_documents()
        logger.info("✅ Document indexing completed successfully.")

        # ✅ Step 2: Configure Streamlit UI
        st.set_page_config(page_title="Financial RAG", layout="wide")
        st.title("📊 Financial RAG Model")
        st.write("Ask financial questions based on real-time company statements.")

        # ✅ Step 3: Setup User Input and Model Response
        user_query = st.text_input("Enter your financial question:")
        if user_query:
            logger.info(f"User Query Received: {user_query}")
            response = rag_pipeline(user_query)
            logger.info("✅ Response generated successfully.")
            st.write("### Model Response:")
            st.write(response)

        # ✅ Step 4: Start Streamlit Server
        port = int(os.getenv("PORT", 8501))  # Default to 8501 if not set
        logger.info(f"Starting Streamlit app on port {port}...")
        st.write(f"🚀 App is running on port: {port}")

    except Exception as e:
        logger.error(f"❌ App failed to start: {e}", exc_info=True)
        st.error("An error occurred while starting the application. Check logs for details.")

# ✅ Run the main function
if __name__ == "__main__":
    main()
