import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# Set RTL direction for Arabic text
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Chroma for local storage
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# Load models (cached after first download)
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    # Use slow tokenizer to avoid conversion error
    tokenizer = AutoTokenizer.from_pretrained('ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA')
    qa_pipeline = pipeline('question-answering', model='ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA', tokenizer=tokenizer)
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

# Process and store Arabic documents
def process_documents(files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    current_time = str(time.time())

    for file in files:
        content = file.read().decode("utf-8")  # Supports Arabic UTF-8
        chunks = text_splitter.split_text(content)
        embeddings = embedder.encode(chunks)
        ids = [f"{current_time}_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
    st.success("تمت معالجة المستندات وإضافتها إلى قاعدة البيانات بنجاح!")

# Streamlit interface
st.title("نظام البحث الذكي في المستندات (عربي)")

# Sidebar for uploading documents
st.sidebar.header("رفع المستندات النصية")
uploaded_files = st.sidebar.file_uploader(
    "ارفع ملفات .txt",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    process_documents(uploaded_files)

# Sidebar for asking questions
st.sidebar.header("اطرح سؤالاً")
question = st.sidebar.text_input("أدخل سؤالك (مثلاً: 'أين الولد؟')")

if question:
    if collection.count() == 0:
        st.error("لم يتم العثور على مستندات. الرجاء رفع ملفات نصية أولاً.")
    else:
        # Generate embedding for the question
        question_embedding = embedder.encode([question])[0]

        # Query Chroma for the most relevant chunk
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=1
        )
        context = results['documents'][0][0]

        # Extract answer using the QA model
        qa_result = qa_pipeline({'question': question, 'context': context})
        answer = qa_result['answer']

        # Display the answer and context
        st.subheader("الإجابة")
        st.write(f"{answer}")

        # Highlight the answer in the context
        start = qa_result['start']
        end = qa_result['end']
        highlighted_context = (
                context[:start] +
                f"**{context[start:end]}**" +
                context[end:]
        )
        st.subheader("السياق المصدر")
        st.markdown(highlighted_context)

# Instructions
st.sidebar.markdown("""
### التعليمات
1. ارفع ملفات `.txt` التي تحتوي على نصوص عربية.
2. انتظر رسالة النجاح التي تؤكد المعالجة.
3. أدخل سؤالاً بالعربية (مثل "أين الولد؟").
4. شاهد الإجابة والسياق من المستند.
""")