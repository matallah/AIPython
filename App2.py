import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# Initialize Chroma client for local storage
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# Load offline models (cached after first download)
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

# Function to process and store uploaded documents
def process_documents(files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    all_embeddings = []
    current_time = str(time.time())

    for file in files:
        content = file.read().decode("utf-8")
        chunks = text_splitter.split_text(content)
        embeddings = embedder.encode(chunks)
        ids = [f"{current_time}_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
    st.success("Documents processed and added to the database successfully!")

# Streamlit interface
st.title("Smart Document Search System")

# Sidebar for document upload
st.sidebar.header("Upload Text Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    process_documents(uploaded_files)

# Sidebar for asking questions
st.sidebar.header("Ask a Question")
question = st.sidebar.text_input("Enter your question (e.g., 'Where is the boy?')")

if question:
    if collection.count() == 0:
        st.error("No documents found. Please upload text files first.")
    else:
        # Generate embedding for the question
        question_embedding = embedder.encode([question])[0]

        # Query Chroma for the most relevant document
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=1
        )
        context = results['documents'][0][0]

        # Extract answer using the QA model
        qa_result = qa_pipeline({'question': question, 'context': context})
        answer = qa_result['answer']

        # Display the answer and context
        st.subheader("Answer")
        st.write(f"{answer}")

        # Highlight the answer in the context
        start = qa_result['start']
        end = qa_result['end']
        highlighted_context = (
                context[:start] +
                f"**{context[start:end]}**" +
                context[end:]
        )
        st.subheader("Source Context")
        st.markdown(highlighted_context)

# Instructions
st.sidebar.markdown("""
### Instructions
1. Upload `.txt` files using the uploader.
2. Wait for the success message confirming processing.
3. Enter a question (e.g., "Where is the boy?") to search.
4. View the answer and the context from the document.
""")