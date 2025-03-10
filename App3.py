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
            cur.execute(
                "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, %s)",
                (sent, emb_list, uploaded_file.name)
            )

def search_database(query, k=5):
    """Search database for similar sentences to query, prioritizing exact matches"""
    # First, check for exact matches
    cur.execute(
        "SELECT content, source FROM documents WHERE content LIKE %s",
        (f"%{query}%",)  # Properly formatting the parameter
    )
    exact_matches = cur.fetchall()

    # If we have enough exact matches, return them
    if len(exact_matches) >= k:
        return [(content, source, 1.0) for content, source in exact_matches[:k]]

    # Otherwise, get vector similarity matches for remaining slots
    query_emb = get_embedding(query)
    remaining = k - len(exact_matches)

    if remaining > 0:
        cur.execute(
            "SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity "
            "FROM documents "
            "WHERE content != %s "
            "ORDER BY similarity DESC "
            "LIMIT %s",
            (query_emb.tolist(), query, remaining)
        )
        similarity_matches = cur.fetchall()
    else:
        similarity_matches = []

    # Combine results, exact matches first
    results = [(content, source, 1.0) for content, source in exact_matches] + \
              [(content, source, similarity) for content, source, similarity in similarity_matches]

    return results[:k]

def display_results(results):
    """Display search results with confidence scores using an enhanced card design."""
    answer_counts = {}

    for idx, (content, source, similarity) in enumerate(results):
        # Reshape the Arabic text for proper display if needed
        reshaped_content = content  # (Apply any reshaping logic if necessary)
        answer = reshaped_content.split(":")[-1].strip() if ":" in reshaped_content else reshaped_content

        # Calculate confidence score in percentage
        confidence = round(similarity * 100, 2)
        source_key = f"{answer} (Source: {source})"

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
        st.markdown(f"""
        <div style="
            text-align: right;
            direction: rtl;
            unicode-bidi: embed;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            background: #f9f9f9;
        ">
            <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 4px;">
                • {answer}
            </div>
            <div style="display: flex; align-items: center;">
                <div style="
                    flex-grow: 1;
                    height: 10px;
                    background: #e0e0e0;
                    border-radius: 5px;
                    margin-left: 8px;
                ">
                    <div style="
                        width: {avg_confidence}%;
                        height: 100%;
                        background: {'#4caf50' if avg_confidence >= 75 else '#ff9800' if avg_confidence >= 50 else '#f44336'};
                        border-radius: 5px;
                    "></div>
                </div>
                <div style="font-size: 0.9em; margin-right: 8px;">
                    {avg_confidence:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
if uploaded_file:
    if st.sidebar.button("Process File"):
        handle_file_upload(uploaded_file)
        st.sidebar.success("File processed successfully!")

# About section
st.sidebar.header("About")
st.sidebar.markdown("""
- **Language Support**: Arabic
- **Embedding Model**: AraBERT (asafaya/bert-base-arabic)
- **Vector Database**: PostgreSQL with PGVector
- **UI Framework**: Streamlit
""")