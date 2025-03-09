from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import numpy as np
import logging

class SmartSearchSystem:
    def __init__(self):
        self.model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
        self.docs_dir = "./your-docs-folder"
        self.persist_dir = "./chroma_db_qwen"
        self.chunk_size = 384

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.llm = self.init_llm()
        self.vector_db = self.init_vector_db()

    def init_llm(self):
        try:
            return LlamaCpp(
                model_path=self.model_path,
                temperature=0.3,
                max_tokens=256,
                n_ctx=2048,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            raise

    def init_vector_db(self):
        try:
            if os.path.exists(self.persist_dir):
                return Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=LlamaCppEmbeddings(model_path=self.model_path)
                )
            return self.create_vector_db()
        except Exception as e:
            self.logger.error(f"Error initializing vector DB: {e}")
            raise

    def create_vector_db(self):
        try:
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
        except Exception as e:
            self.logger.error(f"Error creating vector DB: {e}")
            raise

    def smart_search(self, query):
        try:
            # Retrieve documents with scores
            results = self.vector_db.similarity_search_with_relevance_scores(query, k=5)

            # Calculate threshold and filter documents
            scores = [score for _, score in results]
            threshold = np.percentile(scores, 30) if scores else 0.2
            filtered_docs = [(doc, score) for doc, score in results if score > threshold]

            # Prepare return object
            response_data = {
                "answer": "",
                "results": []
            }

            # Store raw results for frontend
            response_data["results"] = [(doc, float(score)) for doc, score in results]

            if not filtered_docs:
                response_data["answer"] = "No relevant documents found. Try rephrasing your question."
                return response_data

            # Build context with scores
            context_parts = []
            source_set = set()

            for i, (doc, score) in enumerate(filtered_docs, 1):
                source = os.path.basename(doc.metadata['source'])
                content = doc.page_content.replace("\n", " ").strip()
                context_parts.append(f"[Document {i} | {source} (Score: {score:.2f})]\n{content}")
                source_set.add(source)

            context = "\n\n".join(context_parts)

            # Enhanced prompt template
            template = """<|im_start|>system
            You are a document analysis assistant. Base your response strictly on the provided documents.
            Follow these rules:
            1. Use only information from the provided documents
            2. Cite sources using [Source: filename] notation
            3. If unsure, state "This information is not available in the documents"
            4. Keep responses concise and factual<|im_end|>
            <|im_start|>user
            Documents:
            {context}

            Question: {question}<|im_end|>
            <|im_start|>assistant
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Generate response
            chain = LLMChain(llm=self.llm, prompt=prompt)
            answer = chain.run(context=context, question=query)

            # Enforce source citations
            if not any(source in answer for source in source_set):
                answer += f"\n\nSources: {', '.join(source_set)}"

            response_data["answer"] = answer
            return response_data
        except Exception as e:
            self.logger.error(f"Error during smart search: {e}")
            raise

