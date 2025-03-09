import streamlit as st
import chromadb
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import time

# تنسيق النصوص العربية
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# تهيئة قاعدة البيانات
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# تحميل النماذج مع التخزين المؤقت
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    qa_tokenizer = AutoTokenizer.from_pretrained('ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA')
    qa_pipeline = pipeline(
        'question-answering',
        model='ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA',
        tokenizer=qa_tokenizer
    )
    return embedder, qa_pipeline, qa_tokenizer

embedder, qa_pipeline, qa_tokenizer = load_models()  # التصحيح هنا

# دالة لقياس الطول بالتوكينزات
def token_length(text):
    return len(qa_tokenizer.encode(text))

# معالجة المستندات
def process_documents(files):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        qa_tokenizer,
        chunk_size=300,
        chunk_overlap=50,
        separators=['\n\n', '\n', '۔', '؟', '!', '. ', ' '],  # محددات عربية
        is_separator_regex=False  # إضافة هذا الباراميتر
    )

    current_time = str(time.time())

    for file in files:
        content = file.read().decode("utf-8")
        chunks = text_splitter.split_text(content)

        # الفلترة بناءً على التوكينزات
        valid_chunks = [
            chunk for chunk in chunks
            if len(qa_tokenizer.encode(chunk)) <= 512
        ]

        embeddings = embedder.encode(valid_chunks)
        ids = [f"{current_time}_{i}" for i in range(len(valid_chunks))]

        if valid_chunks:
            collection.add(
                documents=valid_chunks,
                embeddings=embeddings,
                ids=ids
            )

    st.success("تمت المعالجة بنجاح!")

# واجهة المستخدم
st.title("نظام البحث الذكي في المستندات (عربي)")

# قسم رفع الملفات
st.sidebar.header("رفع المستندات النصية")
uploaded_files = st.sidebar.file_uploader(
    "ارفع ملفات .txt",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    process_documents(uploaded_files)

# قسم الأسئلة
st.sidebar.header("اطرح سؤالاً")
question = st.sidebar.text_input("أدخل سؤالك (مثلاً: 'أين الولد؟')")

if question:
    if collection.count() == 0:
        st.error("لم يتم العثور على مستندات. الرجاء رفع ملفات نصية أولاً.")
    else:
        # استرجاع السياق
        question_embedding = embedder.encode([question])[0]
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        context = " ".join(results['documents'][0])

        # استخراج الإجابة
        try:
            qa_result = qa_pipeline({'question': question, 'context': context})
            answer = qa_result['answer']

            # عرض النتائج
            st.subheader("الإجابة")
            st.write(f"**{answer}**")

            # تظليل الإجابة في السياق
            start = qa_result['start']
            end = qa_result['end']
            highlighted_context = (
                    context[:start] +
                    f"<mark>{context[start:end]}</mark>" +
                    context[end:]
            )
            st.subheader("السياق المصدر")
            st.markdown(f"<div style='line-height: 2;'>{highlighted_context}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"حدث خطأ في معالجة السؤال: {str(e)}")

# التعليمات
st.sidebar.markdown("""
### التعليمات
1. ارفع ملفات `.txt` مشفرة بـ UTF-8
2. الحد الأقصى لحجم الملف: 10MB
3. الأسئلة المعقدة تحتاج لصياغة واضحة
4. قد تحتاج بعض الإجابات لتحقق يدوي
""")