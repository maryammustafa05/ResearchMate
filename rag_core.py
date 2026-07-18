import chromadb
from pypdf import PdfReader
import io
import cohere
from dotenv import load_dotenv
import os

load_dotenv()
co=cohere.Client(os.getenv("COHERE_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_data")


def read_pdf_bytes(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed_query(query):
    response=co.embed(texts=[query],model="embed-english-v3.0",input_type="search_query")
    return response.embeddings[0]

def embed_texts(texts):
    response=co.embed(texts=texts,model="embed-english-v3.0",input_type="search_query")
    return response.embeddings

def store_chunks(chunks, session_id, paper_title, pdf_url):
    try:
        chroma_client.delete_collection(session_id)
    except Exception:
        pass
    collection = chroma_client.create_collection(session_id)
    embeddings=embed_texts(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadata = [{"chunk_id":i,"title":paper_title,"pdf_url":pdf_url} for i in range (len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )

    return collection