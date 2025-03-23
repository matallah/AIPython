from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import psycopg2
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Initialize database connection
conn = psycopg2.connect(
        dbname='documents_db',
        user='postgres',
        password='postgres',
        host='localhost'
)
cur = conn.cursor()

# Load offline Arabic BERT model
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
        """Generate embedding for text using AraBERT model"""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
                outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def split_arabic_sentences(text):
        """Split Arabic text into sentences using regex"""
        delimiters = r'[.!?]+|[\u06D4\u061F\u060C]+'
        sentences = re.split(delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

def split_and_embed_text(text, doc_id, doc_name):
        """Splits text into chunks, generates embeddings, and prepares data for database insertion."""
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

        embeddings = [get_embedding(chunk) for chunk in chunks]
        data = [(chunk, emb.tolist(), doc_id, doc_name) for chunk, emb in zip(chunks, embeddings)]
        return data

def insert_text_with_metadata(text, doc_id, doc_name):
        """Inserts text with metadata after splitting and embedding."""
        try:
                data = split_and_embed_text(text, doc_id, doc_name)
                with conn:
                        cur.executemany(
                                "INSERT INTO documents (content, embedding, doc_id, doc_name) VALUES (%s, %s, %s, %s)",
                                data
                        )
                return True
        except Exception as e:
                print(f"Error inserting text: {e}")
                return False

def handle_file_upload(uploaded_file):
        """Process uploaded file and store sentences in database"""
        text = uploaded_file.read().decode("utf-8")
        sentences = split_arabic_sentences(text)
        embeddings = [get_embedding(sentence) for sentence in sentences]
        with conn:
                for sent, emb in zip(sentences, embeddings):
                        emb_list = emb.tolist()
                        cur.execute(
                                "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, %s)",
                                (sent, emb_list, uploaded_file.filename)
                        )

def search_database(query, k=15):
        """Search database for similar sentences to query, prioritizing exact matches"""
        cur.execute(
                "SELECT content, source FROM documents WHERE content LIKE %s",
                (f"%{query}%",)
        )
        exact_matches = cur.fetchall()
        if len(exact_matches) >= k:
                return [(content, source, 1.0) for content, source in exact_matches[:k]]

        query_emb = get_embedding(query)
        remaining = k - len(exact_matches)
        if remaining > 0:
                cur.execute(
                        "SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity "
                        "FROM documents "
                        "WHERE content != %s "
                        "ORDER BY similarity DESC "
                        "LIMIT %s",
                        (query_emb.tolist(), query, remaining)
                )
                similarity_matches = cur.fetchall()
        else:
                similarity_matches = []

        results = [(content, source, 1.0) for content, source in exact_matches] + \
                  [(content, source, similarity) for content, source, similarity in similarity_matches]
        return results[:k]

class TextInput(BaseModel):
        text: str
        doc_id: str
        doc_name: str

class SearchQuery(BaseModel):
        query: str
        k: int = 15

@app.post("/insert_text/")
def insert_text(input_data: TextInput):
        """Insert text with metadata into the database."""
        if insert_text_with_metadata(input_data.text, input_data.doc_id, input_data.doc_name):
                return {"message": "Text inserted successfully!"}
        else:
                raise HTTPException(status_code=500, detail="Failed to insert text.")

@app.post("/upload_file/")
def upload_file(file: UploadFile = File(...)):
        """Upload a file and process it to store sentences in the database."""
        handle_file_upload(file)
        return {"message": "File processed successfully!"}

@app.post("/search/")
def search(query_data: SearchQuery):
        """Search the database for similar sentences to the query."""
        results = search_database(query_data.query, query_data.k)
        return {"results": results}

# Run the app with: uvicorn main:app --reload
if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="localhost", port=8000)
# http://localhost:8000/docs