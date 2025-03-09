# doc_search.py
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

class SmartSearchSystem:
    def __init__(self):
        self.model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
        self.docs_dir = "./your-docs-folder"
        self.persist_dir = "./chroma_db_qwen"
        self.chunk_size = 512

        # Initialize components
        self.llm = self.init_llm()
        self.vector_db = self.init_vector_db()

    def init_llm(self):
        return LlamaCpp(
            model_path=self.model_path,
            temperature=0.7,
            max_tokens=512,
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

        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ", ", ". ", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )
        vector_db.persist()
        return vector_db

    def smart_search(self, query):
        results = self.vector_db.similarity_search(query, k=3)

        context = "\n\n".join([
            f"Source: {os.path.basename(doc.metadata['source'])}\nContent: {doc.page_content}"
            for doc in results
        ])

        template = """[INST]
        Analyze these documents and answer the question. Cite sources using [SOURCE] tags.

        Documents:
        {context}

        Question: {question}
        [/INST] Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(context=context, question=query)