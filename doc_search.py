# doc_search.py
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

class SmartSearchSystem:
    def __init__(self):
        self.model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
        self.docs_dir = "./your-docs-folder"
        self.persist_dir = "./chroma_db_qwen"
        self.chunk_size = 384  # Matches the updated splitter config

        # Initialize components
        self.llm = self.init_llm()
        self.vector_db = self.init_vector_db()

    def init_llm(self):
        return LlamaCpp(
            model_path=self.model_path,
            temperature=0.3,  # Start with lower temperature
            max_tokens=256,
            n_ctx=2048,
            verbose=False
        )

    def init_vector_db(self):
        if os.path.exists(self.persist_dir):
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=LlamaCppEmbeddings(model_path=self.model_path)
            )
        return self.create_vector_db()

    def create_vector_db(self):
        embeddings = LlamaCppEmbeddings(
            model_path=self.model_path,
            n_ctx=2048,
            verbose=False
        )

        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", ", "],
            chunk_size=self.chunk_size,
            chunk_overlap=64,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )
        vector_db.persist()
        return vector_db

    import numpy as np

def smart_search(self, query):
    # Retrieve more documents for better selection
    results = self.vector_db.similarity_search_with_relevance_scores(query, k=5)

    # Calculate adaptive threshold
    scores = [score for _, score in results]
    if scores:
        threshold = np.percentile(scores, 30)  # Use 30th percentile as cutoff
    else:
        threshold = 0.2

    filtered_docs = [doc for doc, score in results if score > threshold]

    if not filtered_docs:
        return "No relevant documents found. Try rephrasing your question."

    # Build context with scores
    context_parts = []
    source_set = set()

    for i, (doc, score) in enumerate(filtered_docs, 1):
        source = os.path.basename(doc.metadata['source'])
        content = doc.page_content.replace("\n", " ").strip()
        context_parts.append(f"[Document {i} | {source} (Score: {score:.2f})]\n{content}")
        source_set.add(source)

    # Rest of your code...