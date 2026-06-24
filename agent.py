import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import chromadb
from fastembed import TextEmbedding
import uuid
from rag_core import read_pdf_bytes, chunk_text, store_chunks, embedder, chroma_client

load_dotenv()


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=1,
    timeout=20
)


@tool
def list_available_papers() -> str:
    """Lists all research papers currently available in the system, with their session IDs, titles, and chunk counts."""
    collections = chroma_client.list_collections()
    if not collections:
        return "No papers are currently uploaded."
    
    result = []
    for c in collections:
        collection = chroma_client.get_collection(c.name)
        # Get the first chunk, which usually contains the title
        first_chunk = collection.get(limit=1)
        title_snippet = first_chunk["documents"][0][:150] if first_chunk["documents"] else "Unknown title"
        result.append(f"session_id: {c.name} | chunks: {c.count()} | starts with: {title_snippet}")
    
    return "\n".join(result)
@tool
def ask_paper(session_id: str, question: str) -> str:
    """Answer a question about a specific paper using its session_id. Use this when the user wants information from ONE specific paper."""
    try:
        collection = chroma_client.get_collection(session_id)
    except Exception:
        return f"Error: No paper found with session_id {session_id}"
    
    question_embedding = list(embedder.embed([question]))
    question_embedding = [e.tolist() for e in question_embedding]
    
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)
    
    return f"Relevant context from paper:\n{context}"

@tool
def compare_two_papers(session_id_1: str, session_id_2: str, aspect: str) -> str:
    """Compare two papers on a specific aspect (like methodology, results, or dataset). Use this when the user wants to compare MULTIPLE papers."""
    try:
        collection_1 = chroma_client.get_collection(session_id_1)
        collection_2 = chroma_client.get_collection(session_id_2)
    except Exception:
        return "Error: One or both session_ids not found"
    
    question_embedding = list(embedder.embed([aspect]))
    question_embedding = [e.tolist() for e in question_embedding]
    
    results_1 = collection_1.query(query_embeddings=question_embedding, n_results=2)
    results_2 = collection_2.query(query_embeddings=question_embedding, n_results=2)
    
    # Truncate each chunk to keep total request size manageable
    chunks_1 = [c[:250] for c in results_1["documents"][0]]
    chunks_2 = [c[:250] for c in results_2["documents"][0]]
    context_1 = "\n\n".join(chunks_1)
    context_2 = "\n\n".join(chunks_2)
    
    return f"PAPER A context:\n{context_1}\n\nPAPER B context:\n{context_2}"

@tool
def search_all_papers(query: str) -> str:
    """Search across ALL available papers for a topic or concept, returning which papers mention it. Use this when the user asks something like 'does any paper mention X' or wants to search broadly without specifying a particular paper."""
    collections = chroma_client.list_collections()
    if not collections:
        return "No papers are currently available."
    
    query_embedding = list(embedder.embed([query]))
    query_embedding = [e.tolist() for e in query_embedding]
    
    findings = []
    for c in collections:
        collection = chroma_client.get_collection(c.name)
        results = collection.query(query_embeddings=query_embedding, n_results=1)
        if results["documents"][0]:
            snippet = results["documents"][0][0][:200]
            findings.append(f"Paper {c.name}: {snippet}")
    
    return "\n\n".join(findings)
from arxiv_search import search_arxiv, download_pdf
@tool
def search_and_index_arxiv(topic: str) -> str:
    """Search arXiv for a real paper on a given topic, download it, and index it into the system so it can be asked about. Use this whenever the user wants to FIND a new paper, or asks for papers/suggestions on a topic that isn't already uploaded."""
    results = search_arxiv(topic, max_results=1)
    
    if not results:
        return f"No papers found on arXiv for the topic '{topic}'."
    
    paper = results[0]
    
    if "withdrawn" in paper["title"].lower():
        return f"The top result for '{topic}' was a withdrawn paper. Try rephrasing your search topic."
    
    try:
        pdf_bytes = download_pdf(paper["pdf_url"])
        text = read_pdf_bytes(pdf_bytes)
    except Exception:
        return f"Found a paper titled '{paper['title']}' but couldn't download or read its PDF. It may be unavailable."
    
    if not text.strip():
        return "Found a paper but could not extract its text."
    
    chunks = chunk_text(text)
    session_id = str(uuid.uuid4())
    store_chunks(chunks, session_id)
    
    return f"Found and indexed: \"{paper['title']}\"\nPDF link: {paper['pdf_url']}\nsession_id: {session_id} ({len(chunks)} chunks)\n\nYou can now ask questions about this paper using its session_id."
tools = [list_available_papers, ask_paper, compare_two_papers, search_all_papers,search_and_index_arxiv]

agent_executor = create_agent(llm, tools)

from groq import RateLimitError

if __name__ == "__main__":
    print("ResearchMate Agent ready. Type 'exit' to quit.\n")
    
    from langchain_core.messages import SystemMessage

    conversation_history = [
    ("system", "You are a research assistant. You ONLY know about papers that have been uploaded or indexed into this system — you have NO knowledge of any other papers, including famous ones from your training data. NEVER invent a paper's title or content. If a user asks to find papers on a topic that isn't already available, use the search_and_index_arxiv tool to find and index a REAL paper before answering. Never substitute a well-known paper name you remember from training. When you find or reference a paper, ALWAYS include its real title and PDF link in your response if available — do not omit them even if you think the user only wants the session_id. NEVER generate fake tool results or pretend you called a tool when you did not. If you don't have specific information (like a PDF link) from an actual previous tool call in this conversation, say so honestly — do not search again or invent a new paper unless the user explicitly asks for a different one.")
]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        conversation_history.append(("human", user_input))
        
        try:
            result = agent_executor.invoke({"messages": conversation_history})
            final_message = result["messages"][-1].content
            
            if not final_message or len(final_message.strip()) < 10:
                print("\nAgent: I wasn't able to generate a proper answer for that question. Could you try rephrasing it?\n")
            else:
                print(f"\nAgent: {final_message}\n")
                conversation_history.append(("ai", final_message))
                
        except RateLimitError:
            print("\nAgent: We've hit our usage limit for now. Please try again in a few minutes.\n")
        except Exception as e:
            print(f"\nAgent: Something went wrong while processing your request. Please try again.\n")