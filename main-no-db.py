from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# In-memory storage
DOCUMENTS = []
EMBEDDINGS = []

# Load Arabic BERT model
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def split_arabic_sentences(text: str) -> List[str]:
    """Split Arabic text into sentences using regex"""
    delimiters = r'[.!?]+|[\u06D4\u061F\u060C]+'
    sentences = re.split(delimiters, text)
    return [s.strip() for s in sentences if s.strip()]

def split_and_store_text(text: str, doc_id: str, doc_name: str):
    """Process and store text in memory"""
    sentences = split_arabic_sentences(text)
    max_sentences = 5
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    for chunk in chunks:
        emb = get_embedding(chunk)
        DOCUMENTS.append({
            "content": chunk,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "embedding": emb
        })

def handle_file_upload(uploaded_file: UploadFile):
    """Process uploaded file and store in memory"""
    text = uploaded_file.file.read().decode("utf-8")
    sentences = split_arabic_sentences(text)
    for sentence in sentences:
        emb = get_embedding(sentence)
        DOCUMENTS.append({
            "content": sentence,
            "source": uploaded_file.filename,
            "embedding": emb
        })

def search_documents(query: str, k: int = 15) -> List[Dict]:
    """Search in-memory documents"""
    # Exact matches
    exact_matches = [doc for doc in DOCUMENTS if query.lower() in doc["content"].lower()]

    # Semantic matches if needed
    if len(exact_matches) < k:
        query_emb = get_embedding(query)
        remaining = k - len(exact_matches)

        # Calculate similarities
        similarities = []
        for doc in DOCUMENTS:
            if doc not in exact_matches:
                sim = cosine_similarity(
                    query_emb.reshape(1, -1),
                    doc["embedding"].reshape(1, -1)
                )[0][0]
                similarities.append((doc, sim))

        # Sort by similarity
        semantic_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:remaining]
        semantic_matches = [match[0] for match in semantic_matches]
    else:
        semantic_matches = []

    # Combine results
    results = exact_matches + semantic_matches
    return sorted(results[:k], key=lambda x: x.get("similarity", 1.0), reverse=True)

class TextInput(BaseModel):
    text: str
    doc_id: str
    doc_name: str

class SearchQuery(BaseModel):
    query: str
    k: int = 15

@app.post("/insert_text/")
async def insert_text(input_data: TextInput):
    """Insert text with metadata into memory"""
    try:
        split_and_store_text(input_data.text, input_data.doc_id, input_data.doc_name)
        return {"message": "Text inserted successfully!", "count": len(DOCUMENTS)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process file"""
    try:
        handle_file_upload(file)
        return {"message": "File processed successfully!", "count": len(DOCUMENTS)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
async def search(query_data: SearchQuery):
    """Search documents"""
    try:
        results = search_documents(query_data.query, query_data.k)
        formatted_results = [{
            "content": doc["content"],
            "source": doc.get("source", ""),
            "similarity": doc.get("similarity", 1.0)
        } for doc in results]
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/")
async def get_stats():
    """Get storage statistics"""
    return {
        "document_count": len(DOCUMENTS),
        "embedding_dimension": DOCUMENTS[0]["embedding"].shape[0] if DOCUMENTS else 0
    }

# Run the app with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
# http://localhost:8001/docs