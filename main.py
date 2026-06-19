import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from pypdf import PdfReader
from fastembed import TextEmbedding
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io

load_dotenv()
app=FastAPI( title="ResearchMate API",
            description="Upload a research paper and ask questions about it, with answers grounded in the actual document.",
            version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
groq_client=Groq(api_key=os.getenv("GROQ_API_KEY"))
sessions={}

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
    client = chromadb.Client()
    try:
        client.delete_collection(session_id)
    except Exception:
        pass
    collection = client.create_collection(session_id)

    embeddings = list(embedder.embed(chunks))
    embeddings = [e.tolist() for e in embeddings]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def ask_question(collection, question):
    question_embedding = list(embedder.embed([question]))
    question_embedding = [e.tolist() for e in question_embedding]
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    relevant_chunks = results["documents"][0]

    context = "\n\n".join(relevant_chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "system",
            "content": "You are a research assistant. Answer the question directly and confidently using the provided context. Quote or reference specific details from the context. Only say the information is missing if you genuinely cannot find anything relevant in the context provided."
        }, {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.choices[0].message.content, relevant_chunks


@app.get("/")
def root():
    return {"message":"ResearchMate API is running. Go to /docs to explore the API."}
@app.post("/upload")
async def upload_paper(file:UploadFile=File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    text = read_pdf_bytes(file_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this PDF")

    chunks = chunk_text(text)
    session_id = file.filename.replace(".pdf", "").replace(" ", "_")
    collection = store_chunks(chunks, session_id)
    sessions[session_id] = collection

    return {
        "session_id": session_id,
        "message": f"Paper uploaded and processed into {len(chunks)} chunks",
        "chunk_count": len(chunks)
    }


@app.post("/ask")
def ask(session_id: str, question: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload the paper first.")

    collection = sessions[session_id]
    answer, sources = ask_question(collection, question)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }