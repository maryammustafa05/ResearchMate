import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

embedder = SentenceTransformer('all-mpnet-base-v2')

def read_pdf(path):
    reader = PdfReader(path)
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

def store_chunks(chunks):
    client = chromadb.Client()
    try:
        client.delete_collection("papers")
    except Exception:
        pass
    collection = client.create_collection("papers")

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def ask_question(collection, question, groq_client):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    relevant_chunks = results["documents"][0]

    print("\n--- RETRIEVED CHUNKS ---")
    for i, chunk in enumerate(relevant_chunks):
        print(f"\nChunk {i+1}:\n{chunk[:200]}...")
    print("--- END CHUNKS ---\n")

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


if __name__ == "__main__":
    print("Reading PDF....")
    text = read_pdf("papers/sign_lang.pdf")
    print("Chunking text...")
    chunks = chunk_text(text)
    print("Storing in vector database...")
    collection = store_chunks(chunks)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        answer, sources = ask_question(collection, question, groq_client)
        print(f"\nAnswer: {answer}")
        print(f"\n(Based on {len(sources)} retrieved chunks)\n")