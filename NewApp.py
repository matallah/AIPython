import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Dummy wrapper class for your offline Qwen model.
# Replace this with your actual offline inference code.
class QwenOfflineLLM:
    def __init__(self, model_path):
        self.model_path = model_path
        # Here you would load your offline model from the given path.
    def __call__(self, prompt):
        # Replace the following line with your model's inference
        return f"Offline answer based on prompt: {prompt}"

# Set the path to your offline Qwen2.5-7B-Instruct model
model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"  # <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup><sup data-citation="2" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@Shamimw/building-a-local-rag-based-chatbot-using-chromadb-langchain-and-streamlit-and-ollama-9410559c8a4d">2</a></sup>
llm = QwenOfflineLLM(model_path)  # Initialize offline LLM <sup data-citation="2" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@Shamimw/building-a-local-rag-based-chatbot-using-chromadb-langchain-and-streamlit-and-ollama-9410559c8a4d">2</a></sup>

# Use a sentence-transformer based embeddings model (adjust model name if needed)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup>

# Define a persist directory and collection name for Chroma DB
persist_directory = "./chroma_db"  # <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup>
collection_name = "my_collection"  # <sup data-citation="7" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@vinayjain449/building-an-interactive-document-q-a-system-with-streamlit-and-langchain-586f44575d70">7</a></sup>

st.title("Smart Document Search with Offline Qwen & Chroma")

# Upload TXT files (accepting multiple files)
uploaded_files = st.file_uploader("Upload text documents", type=["txt"], accept_multiple_files=True)  # <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup><sup data-citation="7" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@vinayjain449/building-an-interactive-document-q-a-system-with-streamlit-and-langchain-586f44575d70">7</a></sup>

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode('utf-8')
        # Save source information as metadata
        docs.append({"page_content": content, "metadata": {"source": uploaded_file.name}})

    # Optionally, use a text splitter to chunk large documents into manageable pieces <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup>
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = []
    for doc in docs:
        chunks = text_splitter.split_text(doc["page_content"])
        for chunk in chunks:
            doc_chunks.append({"page_content": chunk, "metadata": doc["metadata"]})

    # Build (or update) the Chroma vector store using the document chunks and metadata <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup><sup data-citation="4" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://dev.to/ngonidzashe/doc-sage-create-a-smart-rag-app-with-langchain-and-streamlit-4lin">4</a></sup>
    vectorstore = Chroma.from_documents(
        documents=[d["page_content"] for d in doc_chunks],
        embedding=embeddings,
        metadatas=[d["metadata"] for d in doc_chunks],
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vectorstore.persist()  # Save the vector store to disk <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup><sup data-citation="4" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://dev.to/ngonidzashe/doc-sage-create-a-smart-rag-app-with-langchain-and-streamlit-4lin">4</a></sup>
    st.success("Documents successfully uploaded and saved!")

    # Input a search query
    query = st.text_input("Enter your search query:")
    if query:
        # Perform similarity search against the locally stored documents <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/76232375/langchain-chroma-load-data-from-vector-database">1</a></sup><sup data-citation="2" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@Shamimw/building-a-local-rag-based-chatbot-using-chromadb-langchain-and-streamlit-and-ollama-9410559c8a4d">2</a></sup>
        docs_found = vectorstore.similarity_search(query, k=5)
        st.markdown("### Search Results")
        for doc in docs_found:
            source = doc.metadata.get("source", "Unknown")
            st.markdown(f"**Document:** {source}")
            st.write(doc.page_content)

        # Optionally build a QA chain that uses the offline Qwen model for enhanced responses <sup data-citation="2" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@Shamimw/building-a-local-rag-based-chatbot-using-chromadb-langchain-and-streamlit-and-ollama-9410559c8a4d">2</a></sup><sup data-citation="7" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://medium.com/@vinayjain449/building-an-interactive-document-q-a-system-with-streamlit-and-langchain-586f44575d70">7</a></sup>
        if st.button("Get Inference Answer"):
            # Create a retriever for the QA chain from the persisted vector store
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            result = qa({"query": query})
            st.markdown("### Offline LLM Answer")
            st.write(result['result'])
