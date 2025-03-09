# app.py
import streamlit as st
from doc_search import SmartSearchSystem  # Assume previous code is in doc_search.py
import os
import time

# Initialize system
@st.cache_resource
def init_system():
    return SmartSearchSystem()

system = init_system()

# Custom CSS
st.markdown("""
<style>
    .stTextInput input {border: 2px solid #4CAF50 !important;}
    .stButton button {background-color: #4CAF50 !important; color: white !important;}
    .doc-box {padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;}
    .answer-box {padding: 20px; background: #f8f9fa; border-radius: 5px; margin: 20px 0;}
    .source-chip {background: #4CAF50; color: white; padding: 2px 8px; border-radius: 15px; font-size: 0.8em;}
</style>
""", unsafe_allow_html=True)

# Sidebar - Document Management
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        doc_dir = "./your-docs-folder"
        os.makedirs(doc_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(doc_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success(f"{len(uploaded_files)} files uploaded!")
        system.vector_db = system.create_vector_db()  # Rebuild index

# Main Interface
st.title("🔍 Smart Document Search")
st.write("Powered by Qwen2.5-7B AI")

# Search Input
query = st.text_input(
    "Enter your question:",
    placeholder="What would you like to search in the documents?",
    key="search_input"
)

if query:
    with st.spinner("🔍 Searching documents..."):
        start_time = time.time()

        # Perform search
        response = system.smart_search(query)

        # Display results
        st.subheader("AI Answer")
        st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)

        # Show performance
        end_time = time.time()
        st.caption(f"⏱️ Response time: {end_time - start_time:.2f}s")

# Display current documents
st.sidebar.markdown("---")
st.sidebar.subheader("Indexed Documents")
if os.path.exists("./your-docs-folder"):
    docs = [f for f in os.listdir("./your-docs-folder") if f.endswith((".txt", ".md"))]
    for doc in docs:
        st.sidebar.markdown(f'<div class="doc-box">📄 {doc}</div>', unsafe_allow_html=True)
else:
    st.sidebar.warning("No documents uploaded yet")