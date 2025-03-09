import os
import logging

import numpy as np
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from gensim.models import KeyedVectors
from transformers import pipeline

from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Correct and update the model paths
model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
embeddings_model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/embeddings"

# Define document collection
docs_dir = "./your-docs-folder"
loader = DirectoryLoader(docs_dir, ["*.txt"])
documents = loader.load()

# Initialize Whoosh index
index_directory = "./whoosh_index"
index = create_in(index_directory, Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT))

# Functions to handle embeddings and model loading
def load_embeddings_model(model_path):
    try:
        logging.info(f"Attempting to load embeddings model from: {model_path}")
        embeddings_model = LlamaCppEmbeddings(model_path=model_path, n_ctx=2048, verbose=True)
        logging.info("Loading successful.")
        return embeddings_model
    except FileNotFoundError:
        logging.error(f"Model not found at path: {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading embeddings model: {e}")
        raise

# Usage in script
logging.basicConfig(level=logging.INFO)
embeddings_model = load_embeddings_model(embeddings_model_path)


def get_llm_embedding(text):
    # Simplified method assuming text preprocessing and tokenization are handled elsewhere
    return embeddings_model.embed_text(text)

# Hybrid search function
def hybrid_smart_search(query, ix, top_n_ir=10, top_n_llm=5):
    with ix.searcher() as searcher:
        ir_results = searcher.search(query, limit=top_n_ir)

    top_ir_docs = [doc for _, doc in ir_results]

    # Query embedding using the correct Llama2 embeddings model
    query_embedding = get_llm_embedding(query)

    doc_embeddings = [get_llm_embedding(doc.content) for doc in top_ir_docs]

    refined_scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

    hybrid_scores = {doc: (refined_scores[i] * 0.6 + ir_results[doc.path][0] * 0.4) for i, doc in enumerate(top_ir_docs)}

    ranked_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_llm]

    return ranked_docs

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def prepare_response(ranked_docs, query):
    response_data = {
        "answer": "",
        "results": []
    }

    context = "\n\n".join([f"[Document {i+1} | {os.path.basename(doc.path)} (Score: {score:.2f})]\n{doc.content}" for i, (doc, score) in enumerate(ranked_docs)])

    # Create prompt with LLMChain
    chain = LLMChain(llm=LlamaCpp(model_path=model_path), prompt=f"""
        system
        You are a document analysis assistant. Base your response strictly on the provided documents.
        Follow these rules:
        1. Use only information from the provided documents
        2. Cite sources using [Source: filename] notation
        3. If unsure, state "This information is not available in the documents"
        4. Keep responses concise and factual

        user
        Documents:
        {context}

        Question: {query}
    """)

    answer = chain.run(context=context, question=query)

    sources_used = [os.path.basename(doc.path) for _, doc in ranked_docs]
    if not any(source in answer for source in sources_used):
        answer += f"\n\nSources: {', '.join(sources_used)}"

    response_data["answer"] = answer
    return response_data

# Example usage
if __name__ == "__main__":
    query = "Your search query here"
    response = prepare_response(hybrid_smart_search(query, index), query)
    print(response)
