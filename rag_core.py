import chromadb
from fastembed import TextEmbedding
from pypdf import PdfReader
import io

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
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


def store_chunks(chunks, session_id):
    try:
        chroma_client.delete_collection(session_id)
    except Exception:
        pass
    collection = chroma_client.create_collection(session_id)
    embeddings = list(embedder.embed(chunks))
    embeddings = [e.tolist() for e in embeddings]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection