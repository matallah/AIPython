import streamlit as st
import requests

# Define the base URL for the FastAPI app
BASE_URL = "http://127.0.0.1:8000"

def display_results(results):
    """Display search results with confidence scores using an enhanced card design."""
    answer_counts = {}

    for idx, (content, source, similarity) in enumerate(results):
        reshaped_content = content  # Apply any reshaping logic if necessary
        answer = reshaped_content.split(":")[-1].strip() if ":" in reshaped_content else reshaped_content
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

st.title("Semantic Search")

# Question section
query = st.text_input("Ask a question about the documents:", "")
if query:
    with st.spinner("Searching documents..."):
        response = requests.post(f"{BASE_URL}/search/", json={"query": query})
        results = response.json().get("results", [])
        display_results(results)

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload Arabic documents in TXT, PDF, or DOCX format
2. Ask questions in Arabic
3. View answers with confidence scores
4. Answers are retrieved from the uploaded documents
""")

# Direct Text Input Section
st.sidebar.header("Insert Text Directly")
text_input = st.sidebar.text_area("Enter text:")
doc_id_input = st.sidebar.text_input("Document ID:")
doc_name_input = st.sidebar.text_input("Document Name:")

if st.sidebar.button("Insert Text"):
    if text_input and doc_id_input and doc_name_input:
        response = requests.post(f"{BASE_URL}/insert_text/", json={"text": text_input, "doc_id": doc_id_input, "doc_name": doc_name_input})
        if response.status_code == 200:
            st.sidebar.success("Text inserted successfully!")
        else:
            st.sidebar.error("Failed to insert text.")
    else:
        st.sidebar.warning("Please fill in all fields.")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt"])
if uploaded_file:
    if st.sidebar.button("Process File"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{BASE_URL}/upload_file/", files=files)
        if response.status_code == 200:
            st.sidebar.success("File processed successfully!")
        else:
            st.sidebar.error("Failed to process file.")

# About section
st.sidebar.header("About")
st.sidebar.markdown("""
- **Language Support**: Arabic
- **Embedding Model**: AraBERT (asafaya/bert-base-arabic)
- **Vector Database**: PostgreSQL with PGVector
- **UI Framework**: Streamlit
""")
