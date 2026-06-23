import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import chromadb
from fastembed import TextEmbedding

load_dotenv()

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_data")

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
tools = [list_available_papers, ask_paper, compare_two_papers, search_all_papers]

agent_executor = create_agent(llm, tools)

from groq import RateLimitError

if __name__ == "__main__":
    print("ResearchMate Agent ready. Type 'exit' to quit.\n")
    
    from langchain_core.messages import SystemMessage

    conversation_history = [
    ("system", "You are a research assistant. You ONLY know about papers that have been uploaded into this system — you have NO knowledge of any other papers, including famous ones like 'Attention is All You Need' or 'BERT'. NEVER invent or assume a paper's title, author, or content. If you don't have a title from your tools, refer to the paper only by its session_id or by its actual retrieved content. If a user asks for 'the best' or 'most relevant' paper, you must look at the actual available papers' real content first — never default to well-known paper names you remember from training.")
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