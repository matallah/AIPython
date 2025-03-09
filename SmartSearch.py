import sys

from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate  # Missing import
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import contextlib

# Add suppression for llama.cpp output
@contextlib.contextmanager
def suppress_llama_output():
    with open(os.devnull, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = original_stdout

model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"

# Configuration
DOCS_DIR = "./your-docs-folder"
PERSIST_DIR = "./chroma_db_qwen"
CHUNK_SIZE = 512

class CustomTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        separators = ["\n\n", "\n", ", ", ". ", " ", ""]
        super().__init__(separators=separators, **kwargs)

def create_vector_db():
    embeddings = LlamaCppEmbeddings(
        model_path=model_path,
        n_ctx=2048,
        verbose=False
    )

    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt")
    documents = loader.load()

    text_splitter = CustomTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vector_db.persist()
    return vector_db

def smart_search(query, vector_db, llm, k=3):
    results = vector_db.similarity_search(query, k=k)

    context = "\n\n".join([
        f"Source: {os.path.basename(doc.metadata['source'])}\nContent: {doc.page_content}"
        for doc in results
    ])

    template = """
    You are a research assistant. Analyze the following documents and answer the question.
    Include source references in your answer.

    Documents:
    {context}

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, question=query)

# Initialize LLM with suppressed output
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    verbose=False
)

if __name__ == "__main__":
    if not os.path.exists(PERSIST_DIR):
        print("Creating new vector database...")
        with suppress_llama_output():
            vector_db = create_vector_db()
    else:
        print("Loading existing database...")
        with suppress_llama_output():
            embeddings = LlamaCppEmbeddings(model_path=model_path)
            vector_db = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )

    while True:
        query = input("\nEnter your search query (or 'exit'): ").strip()
        if query.lower() in ['exit', 'quit']:
            break

        try:
            print("\nProcessing...", end='', flush=True)
            with suppress_llama_output():
                response = smart_search(query, vector_db, llm)
            print("\r" + " "*20 + "\r")  # Clear processing message
            print(f"Answer: {response}\n")
        except Exception as e:
            print(f"\nError: {str(e)}")