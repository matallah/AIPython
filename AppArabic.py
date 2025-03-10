import os
import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import ast
import streamlit as st
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import numpy as np
from pyarabic.araby import normalize_hamza

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
SUPPORTED_FILE_TYPE = '.txt'
EMBEDDING_DIM = 768
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Arabic stopwords
ARABIC_STOPWORDS = {
    "في", "من", "إلى", "على", "أن", "إن", "اين", "لا", "ما", "ماذا", "هل", "أين",
}

@dataclass
class Config:
    db_params: dict = None
    def __post_init__(self):
        load_dotenv()
        self.db_params = {
            "dbname": os.getenv("DB_NAME", "documents_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }

class DatabaseManager:
    def __init__(self, config: Config):
        self.conn = None
        self.cur = None
        self.config = config

    def connect(self) -> None:
        try:
            self.conn = psycopg2.connect(**self.config.db_params)
            self.cur = self.conn.cursor()
            self._setup_schema()
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            raise

    def _setup_schema(self) -> None:
        try:
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
        except Exception as e:
            logging.warning(f"Vector extension setup failed: {str(e)}")
            self.conn.rollback()

        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL
            )
        """)
        self.conn.commit()

    def close(self) -> None:
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

class DocumentProcessor:
    def __init__(self, db_manager: DatabaseManager, embedder: SentenceTransformer, tokenizer: AutoTokenizer):
        self.db_manager = db_manager
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '۔', '؟', '!', '. ', ' ']
        )

    def validate_files(self, files: List) -> List:
        valid_files = []
        for file in files:
            if file.size > MAX_FILE_SIZE:
                st.error(f"ملف {file.name} كبير جداً. الحد الأقصى 5MB")
                continue
            if not file.name.endswith(SUPPORTED_FILE_TYPE):
                st.error(f"تنسيق الملف {file.name} غير مدعوم")
                continue
            valid_files.append(file)
        return valid_files

    def process_files(self, files: List) -> int:
        total_chunks = 0
        for file in files:
            try:
                content = file.read().decode("utf-8")
                content = normalize_hamza(content)
                chunks = self.text_splitter.split_text(content)
                valid_chunks = [c for c in chunks if len(self.tokenizer.encode(c)) <= 512]
                if valid_chunks:
                    embeddings = self.embedder.encode(valid_chunks, show_progress_bar=False)
                    data = [(chunk, embedding.tolist()) for chunk, embedding in zip(valid_chunks, embeddings)]
                    execute_values(
                        self.db_manager.cur,
                        "INSERT INTO documents (text, embedding) VALUES %s",
                        data,
                        template="(%s, %s::vector)"
                    )
                    self.db_manager.conn.commit()
                    total_chunks += len(data)
            except Exception as e:
                logging.error(f"Error processing {file.name}: {str(e)}")
                st.error(f"خطأ في معالجة {file.name}. تأكد من ترميز UTF-8")
        return total_chunks

class SearchEngine:
    def __init__(self, db_manager: DatabaseManager, embedder: SentenceTransformer, qa_pipeline: pipeline):
        self.db_manager = db_manager
        self.embedder = embedder
        self.qa_pipeline = qa_pipeline

    def extract_keywords(self, question: str) -> List[str]:
        cleaned = re.sub(r'[؟?،,.]', '', question)
        words = [word for word in cleaned.split()
                 if word not in ARABIC_STOPWORDS and len(word) > 2]
        return list(set(words))

    def search(self, question: str) -> Tuple[str, str]:
        keywords = self.extract_keywords(question)
        question_embedding = self.embedder.encode([question])[0]
        chunks = self._get_candidate_chunks(keywords, question_embedding)

        if not chunks:
            return "لا توجد نتائج كافية", ""

        context = " ".join(chunk[0] for chunk in chunks[:3])
        try:
            qa_result = self.qa_pipeline(question=question, context=context)
            start, end = qa_result['start'], qa_result['end']
            highlighted = f"{context[:start]}<mark>{context[start:end]}</mark>{context[end:]}"
            return qa_result['answer'], highlighted
        except Exception as e:
            logging.error(f"QA Error: {str(e)}")
            return "تعذر إيجاد إجابة واضحة", context

    def _get_candidate_chunks(self, keywords: List[str], question_embedding: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        chunks = []

        # Keyword search
        for keyword in keywords[:3]:
            self.db_manager.cur.execute("""
                SELECT text, embedding 
                FROM documents 
                WHERE text ~* %s 
                LIMIT 3
            """, (keyword,))
            chunks.extend(self.db_manager.cur.fetchall())

        # Semantic search fallback
        if len(chunks) < 3:
            self.db_manager.cur.execute("""
                SELECT text, embedding 
                FROM documents 
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
            """, (question_embedding.tolist(), 5 - len(chunks)))
            chunks.extend(self.db_manager.cur.fetchall())

        # Rank chunks
        scored_chunks = []
        for text, embedding_str in chunks:
            try:
                embedding = np.array(ast.literal_eval(embedding_str), dtype=np.float32)
                keyword_score = sum(text.count(kw) for kw in keywords)
                similarity = np.dot(embedding, question_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(question_embedding)
                )
                scored_chunks.append((text, embedding, keyword_score + similarity * 2))
            except Exception as e:
                logging.error(f"Error processing chunk: {str(e)}")
                continue

        return sorted(scored_chunks, key=lambda x: x[2], reverse=True)

def load_models():
    @st.cache_resource
    def _load():
        embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        tokenizer = AutoTokenizer.from_pretrained('ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA')
        qa_pipeline = pipeline(
            'question-answering',
            model='ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA',
            tokenizer=tokenizer
        )
        return embedder, tokenizer, qa_pipeline
    return _load()

def main():
    st.set_page_config(page_title="نظام البحث الذكي", layout="wide")
    config = Config()
    db_manager = DatabaseManager(config)
    db_manager.connect()
    embedder, tokenizer, qa_pipeline = load_models()
    processor = DocumentProcessor(db_manager, embedder, tokenizer)
    search_engine = SearchEngine(db_manager, embedder, qa_pipeline)

    # Custom CSS
    st.markdown("""
        <style>
        body { 
            direction: rtl; 
            text-align: right; 
            font-family: 'Lateef', sans-serif; 
            max-width: 1200px;
            margin: 0 auto;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 2rem 0;
        }
        .search-box {
            width: 60%;
            margin-bottom: 1.5rem;
        }
        .result-container {
            width: 80%;
            margin: 0 auto;
            text-align: right;
        }
        .ar-text { 
            line-height: 2; 
            padding: 1rem; 
            text-align: justify;
        }
        mark { 
            background-color: #ffff00; 
            padding: 0.2rem; 
        }
        .sidebar .sidebar-content { 
            background-color: #f5f5f5; 
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("نظام البحث الذكي في المستندات العربية")

    # Main content area
    with st.container():
        # Search section
        with st.form(key='search_form'):
            col1, col2 = st.columns([6, 1])
            with col1:
                question = st.text_input(
                    "أدخل سؤالك هنا",
                    placeholder="مثال: متى تم افتتاح الجامعة؟",
                    key="search_query"
                )
            with col2:
                st.write("")  # Vertical alignment
                st.write("")
                submit_button = st.form_submit_button("بحث")

        # Results section
        if submit_button and question.strip():
            with st.spinner("البحث عن الإجابة..."):
                db_manager.cur.execute("SELECT COUNT(*) FROM documents")
                if db_manager.cur.fetchone()[0] == 0:
                    st.error("⚠️ لم يتم تحميل أي مستندات بعد!")
                else:
                    answer, context = search_engine.search(question)
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.subheader("الإجابة:")
                    st.markdown(f"<div class='ar-text'>{answer}</div>", unsafe_allow_html=True)
                    st.subheader("السياق المرجعي:")
                    st.markdown(f"<div class='ar-text'>{context}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar for uploads
    with st.sidebar:
        st.header("إدارة المستندات")
        uploaded_files = st.file_uploader(
            "رفع ملفات TXT (UTF-8)",
            type=["txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            valid_files = processor.validate_files(uploaded_files)
            if valid_files:
                with st.spinner("جارٍ معالجة المستندات..."):
                    count = processor.process_files(valid_files)
                    st.success(f"تمت معالجة {count} قطعة نصية بنجاح!")
        st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <p>طور بواسطة <a href="https://example.com" target="_blank">فريق الذكاء الاصطناعي</a></p>
            </div>
        """, unsafe_allow_html=True)

    db_manager.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        st.error("حدث خطأ في تشغيل التطبيق")