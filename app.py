
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

        # Display main answer
        st.subheader("AI Answer")
        st.markdown(f'<div class="answer-box">{response["answer"]}</div>',
                    unsafe_allow_html=True)

        # Display raw results
        st.subheader("Search Results Analysis")

        cols = st.columns([1, 4])
        with cols[0]:
            min_score = st.slider(
                "Filter by confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05
            )

        with cols[1]:
            st.caption(f"Showing {len(response['results'])} total matches")

        for i, (doc, score) in enumerate(response["results"], 1):
            if score < min_score:
                continue

            source = os.path.basename(doc.metadata['source'])
            content = doc.page_content.replace("\n", " ").strip()

            with st.expander(f"Match {i}: {source} (Score: {score:.2f})"):
                col1, col2 = st.columns([2, 8])
                with col1:
                    st.metric("Confidence", f"{score:.2f}")
                    st.progress(score)
                with col2:
                    st.write(content)

        # Performance metrics
        end_time = time.time()
        st.caption(f"⏱️ Response time: {end_time - start_time:.2f}s")

# Display current documents with content preview
st.sidebar.markdown("---")
st.sidebar.subheader("Indexed Documents")

if os.path.exists("./your-docs-folder"):
    docs = [f for f in os.listdir("./your-docs-folder") if f.endswith((".txt", ".md"))]

    # Initialize session state for selected document
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None

    for doc in docs:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            # Create clickable document names
            if st.button(f"📄 {doc}", key=f"doc_{doc}"):
                st.session_state.selected_doc = doc
        with col2:
            # Show document size
            doc_path = os.path.join("./your-docs-folder", doc)
            size = os.path.getsize(doc_path) / 1024  # KB
            st.caption(f"{size:.1f}KB")

    # Display selected document content
    if st.session_state.selected_doc:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Document Preview")
        try:
            doc_path = os.path.join("./your-docs-folder", st.session_state.selected_doc)
            with open(doc_path, "r") as f:
                content = f.read()

            st.sidebar.text_area(
                "File Content",
                value=content,
                height=300,
                key=f"content_{st.session_state.selected_doc}",
                disabled=True
            )
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
else:
    st.sidebar.warning("No documents uploaded yet")

# What is the special code?