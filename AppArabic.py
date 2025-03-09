import streamlit as st
import chromadb
import re
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from collections import defaultdict

# تنسيق النصوص العربية
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .ar-text {
        line-height: 2;
    }
    mark {
        background-color: yellow;
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

embedder, qa_pipeline, qa_tokenizer = load_models()

# استخراج الكلمات المفتاحية
def extract_keywords(question):
    arabic_stopwords = {"في", "من", "إلى", "على", "أن", "إن", "أن", "قد", "لا", "ما", "ماذا", "هل", "أين"}
    question = re.sub(r'[؟?،,.]', '', question)
    return [word for word in question.split() if word not in arabic_stopwords and len(word) > 2]

# معالجة المستندات
def process_documents(files):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        qa_tokenizer,
        chunk_size=300,
        chunk_overlap=50,
        separators=['\n\n', '\n', '۔', '؟', '!', '. ', ' '],
        is_separator_regex=False
    )

    current_time = str(time.time())

    for file in files:
        content = file.read().decode("utf-8")
        chunks = text_splitter.split_text(content)

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

    st.success("تمت معالجة المستندات بنجاح!")

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
        keywords = extract_keywords(question)
        all_chunks = []

        # البحث عن كل كلمة مفتاحية
        if keywords:
            for keyword in keywords:
                results = collection.get(
                    where={"document": {"$eq": keyword}},
                    limit=3
                )
                if results['documents']:
                    all_chunks.extend(results['documents'][0])

        # إذا لم توجد نتائج دقيقة
        if not all_chunks:
            question_embedding = embedder.encode([question])[0]
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=5
            )
            all_chunks = results['documents'][0]

        # نظام التصويت للقطع
        chunk_scores = defaultdict(int)
        for chunk in all_chunks:
            for keyword in keywords:
                chunk_scores[chunk] += chunk.count(keyword)

        # ترتيب النتائج
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        final_chunks = [chunk for chunk, score in sorted_chunks[:3]]

        # التحضير للعرض
        context = " ".join(final_chunks)

        try:
            qa_result = qa_pipeline({'question': question, 'context': context})
            answer = qa_result['answer']

            # عرض الإجابة
            st.subheader("الإجابة")
            st.markdown(f"<div class='ar-text'>**{answer}**</div>", unsafe_allow_html=True)

            # تظليل الإجابة في السياق
            start = qa_result['start']
            end = qa_result['end']
            highlighted_context = (
                    context[:start] +
                    f"<mark>{context[start:end]}</mark>" +
                    context[end:]
            )
            st.subheader("السياق المصدر")
            st.markdown(f"<div class='ar-text'>{highlighted_context}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"حدث خطأ في معالجة السؤال: {str(e)}")

# التعليمات
st.sidebar.markdown("""
### التعليمات
1. ارفع ملفات نصية عربية فقط (UTF-8)
2. استخدم أسئلة مباشرة تحتوي على كلمات مفتاحية
3. الأسماء والأماكن تحصل على أولوية في النتائج
4. الحد الأقصى لحجم الملف: 5MB
""")