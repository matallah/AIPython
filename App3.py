import streamlit as st
import psycopg2
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from arabic_reshaper import reshape
import bidi.algorithm

# Initialize database connection
conn = psycopg2.connect(
    dbname='documents_db',
    user='postgres',
    password='postgres',
    host='localhost'
)
cur = conn.cursor()

# Load offline Arabic BERT model
model_name = "asafaya/bert-base-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    """Generate embedding for text using AraBERT model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def reshape_arabic_text(text):
    """Reshape Arabic text for proper display"""
    reshaped_text = reshape(text)
    return bidi.algorithm.get_display(reshaped_text)

def split_arabic_sentences(text):
    """Split Arabic text into sentences using regex"""
    # Arabic sentence delimiters: .!? and Arabic-specific characters
    delimiters = r'[.!?]+|[\u06D4\u061F\u060C]+'
    sentences = re.split(delimiters, text)
    return [s.strip() for s in sentences if s.strip()]

def handle_file_upload(uploaded_file):
    """Process uploaded file and store sentences in database"""
    text = uploaded_file.read().decode("utf-8")
    sentences = split_arabic_sentences(text)

    # Generate embeddings for each sentence
    embeddings = [get_embedding(sentence) for sentence in sentences]

    # Insert into database
    with conn:
        for sent, emb in zip(sentences, embeddings):
            emb_list = emb.tolist()  # Convert embedding to list
            print(f"Inserting embedding: {emb_list}")  # Debug print
            cur.execute(
                "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, %s)",
                (sent, emb_list, uploaded_file.name)
            )



def search_database(query, k=5):
    """Search database for similar sentences to query"""
    query_emb = get_embedding(query)
    cur.execute(
        "SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity "
        "FROM documents "
        "ORDER BY similarity DESC "
        "LIMIT %s",
        (query_emb.tolist(), k)  # Ensure embedding is passed as a list and cast to vector
    )
    return cur.fetchall()



def display_results(results):
    """Display search results with confidence scores"""
    answer_counts = {}

    for idx, (content, source, similarity) in enumerate(results):
        answer = content.split(":")[-1].strip() if ":" in content else content
        reshaped_answer = reshape_arabic_text(answer)

        # Calculate confidence score
        confidence = round(similarity * 100, 2)
        source_key = f"{reshaped_answer} (Source: {source})"

        if source_key in answer_counts:
            answer_counts[source_key].append(confidence)
        else:
            answer_counts[source_key] = [confidence]

    st.subheader("Answers:")
    if not answer_counts:
        st.write("No relevant information found.")
        return

    for answer, confidences in answer_counts.items():
        avg_confidence = sum(confidences) / len(confidences)
        st.write(f"• {answer} (Confidence: {avg_confidence}%)")

# Streamlit UI
st.title("Arabic Document QA System")
st.markdown("Upload documents and ask questions about their content.")

# File upload section
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
if uploaded_file:
    if st.button("Process File"):
        handle_file_upload(uploaded_file)
        st.success("File processed successfully!")

# Question section
query = st.text_input("Ask a question about the documents:", "")
if query:
    with st.spinner("Searching documents..."):
        results = search_database(query)
        display_results(results)

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload Arabic documents in TXT, PDF, or DOCX format
2. Ask questions in Arabic
3. View answers with confidence scores
4. Answers are retrieved from the uploaded documents
""")

# About section
st.sidebar.header("About")
st.sidebar.markdown("""
- **Language Support**: Arabic
- **Embedding Model**: AraBERT (asafaya/bert-base-arabic)
- **Vector Database**: PostgreSQL with PGVector
- **UI Framework**: Streamlit
""")