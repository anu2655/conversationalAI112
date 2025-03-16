import os
import streamlit as st
import logging
from your_module import index_documents, rag_pipeline  # Ensure these imports are correct

# ✅ Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        # ✅ Step 1: Index Documents Before Launching Streamlit
        logger.info("Starting document indexing...")
        index_documents()
        logger.info("✅ Document indexing completed successfully.")

        # ✅ Step 2: Configure Streamlit UI
        st.set_page_config(page_title="Financial RAG", layout="wide")
        st.title("Financial RAG Model")
        st.write("Ask financial questions based on company statements.")

        # ✅ Step 3: Setup User Input and Model Response
        user_query = st.text_input("Enter your financial question:")
        if user_query:
            logger.info(f"User Query Received: {user_query}")
            response = rag_pipeline(user_query)
            logger.info("✅ Response generated successfully.")
            st.write("### Model Response:")
            st.write(response)

        # ✅ Step 4: Start Streamlit Server
        if __name__ == "__main__":
            port = int(os.getenv("PORT", 8501))  # Default to 8501 if not set
            logger.info(f"Starting Streamlit app on port {port}...")
            st.write(f"🚀 App is running on port: {port}")

    except Exception as e:
        logger.error(f"❌ App failed to start: {e}", exc_info=True)
        st.error("An error occurred while starting the application. Check logs for details.")

# ✅ Run the main function
if __name__ == "__main__":
    main()
