CURRENT_SESSION_ID = None
CURRENT_PAPER_TITLE = None
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
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=1,
    timeout=20
)
@tool
def check_citation(session_id: str, claim: str) -> str:
    """Verify whether a specific claim or citation is actually supported by a paper's content. Use this when a user wants to fact-check a sentence they wrote against a source paper, NOT for general questions about the paper."""
    try:
        collection = chroma_client.get_collection(session_id)
    except Exception:
        return f"Error: No paper found with session_id {session_id}"
    
    claim_embedding = list(embedder.embed([claim]))
    claim_embedding = [e.tolist() for e in claim_embedding]
    
    results = collection.query(query_embeddings=claim_embedding, n_results=3)
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)
    
    return f"CLAIM TO VERIFY: {claim}\n\nMOST RELEVANT PAPER CONTENT FOUND:\n{context}\n\nBased on this content, judge whether the claim is: SUPPORTED (matches the paper), CONTRADICTED (paper says something different), or NOT FOUND (paper doesn't address this). Be specific about what the paper actually says if there's a discrepancy."
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
    """
    Answer a question about a specific paper using its session_id.
    Use this when the user wants information from ONE specific paper.
    """
    print("===== ASK_PAPER CALLED =====")
    print("SESSION ID RECEIVED:", session_id)
    try:
        collection = chroma_client.get_collection(session_id)
    except Exception:
        return f"Error: No paper found with session_id {session_id}"

    # Embed user question
    question_embedding = list(embedder.embed([question]))
    question_embedding = [e.tolist() for e in question_embedding]
    print("COLLECTION COUNT:", collection.count())
    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=5
    )
    print("\n===== RETRIEVED RESULTS =====")
    print(results)

    if not results.get("documents") or not results["documents"][0]:
        return "No relevant information found in the paper."

    chunks = results["documents"][0]

    # Metadata may not exist for older collections
    metadata_results = results.get("metadatas")

    if metadata_results and len(metadata_results) > 0:
        metadata_list = metadata_results[0]
    else:
        metadata_list = []

    response_parts = []

    # Default paper info
    paper_title = "Unknown Title"
    pdf_url = "Unknown PDF"

    # Safely extract title and PDF
    if (
        metadata_list
        and len(metadata_list) > 0
        and metadata_list[0] is not None
    ):
        paper_title = metadata_list[0].get(
            "title",
            "Unknown Title"
        )

        pdf_url = metadata_list[0].get(
            "pdf_url",
            "Unknown PDF"
        )

    response_parts.append(
        f"""PAPER TITLE:
         {paper_title}

        PDF:
        {pdf_url}
        """
    )

    response_parts.append(
        f"""
    USER QUESTION:
         {question}

    RETRIEVED EVIDENCE:
    """
    )

    # Add retrieved chunks
    for i, chunk in enumerate(chunks):

        chunk_id = "Unknown"

        if (
            metadata_list
            and i < len(metadata_list)
            and metadata_list[i] is not None
        ):
            chunk_id = metadata_list[i].get(
                "chunk_id",
                "Unknown"
            )

        response_parts.append(
            f"""
        ----- SOURCE CHUNK {i + 1} -----
        Chunk ID: {chunk_id}

         {chunk}
           """
        )

    response_parts.append(
        """
INSTRUCTIONS FOR THE ASSISTANT:

- Answer ONLY using the retrieved evidence above.
- Do NOT use outside knowledge.
- Do NOT guess.
- If the evidence is insufficient, say:
  "The retrieved sections do not contain enough information to answer that question."
- Cite which source chunks support your answer.
"""
    )
    final_response = "\n".join(response_parts)

    print("\n===== TOOL RETURN =====")
    print(final_response[:3000])

    return "\n".join(response_parts)

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
    """
    MANDATORY TOOL.

    Call this tool whenever the user:
    - asks for a paper
    - asks for a research paper
    - asks for a paper recommendation
    - asks for a PDF link
    - asks for an NLP paper
    - asks for an AI paper
    - asks to search arXiv

    Never answer such requests without calling this tool first."""
    
    results = search_arxiv(topic, max_results=1)
    
    if not results:
        return f"No papers found on arXiv for the topic '{topic}'."
    
    paper = results[0]
    
    # Check if we already have a paper with this exact title indexed
    existing_collections = chroma_client.list_collections()
    for c in existing_collections:
        collection = chroma_client.get_collection(c.name)
        first_chunk = collection.get(limit=1)
        if first_chunk["documents"] and paper["title"][:50].lower() in first_chunk["documents"][0][:300].lower():
            return f"This paper is already indexed: \"{paper['title']}\"\nPDF link: {paper['pdf_url']}\nsession_id: {c.name}\n\nYou can ask questions about it using this session_id."
    
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
    global CURRENT_SESSION_ID
    global CURRENT_PAPER_TITLE

    CURRENT_SESSION_ID = session_id
    CURRENT_PAPER_TITLE = paper["title"]
    store_chunks(chunks,
    session_id,
    paper["title"],
    paper["pdf_url"])
    
    return f"Found and indexed paper: Title: {paper['title']} PDF: {paper['pdf_url']}, You can now ask questions about this paper."
@tool
def find_session_by_name(paper_name: str) -> str:
    """Find a paper's session_id by matching part of its name or title. Use this internally when a user refers to a paper by name instead of session_id."""
    print("===== FIND_SESSION_BY_NAME CALLED =====")
    collections = chroma_client.list_collections()
    for c in collections:
        collection = chroma_client.get_collection(c.name)
        first_chunk = collection.get(limit=1)
        meta = first_chunk.get("metadatas")

        if meta and meta[0]:
           title_snippet = meta[0].get("title", "")
        else:
           title_snippet = ""
        if paper_name.lower() in title_snippet.lower():
            return c.name
    return None
@tool
def ask_current_paper(question: str) -> str:
    """
    Ask a question about the currently selected paper.
    """
    global CURRENT_SESSION_ID
    print("===== ASK_CURRENT_PAPER CALLED =====")
    print("CURRENT_SESSION_ID =", CURRENT_SESSION_ID)
    print("QUESTION =", question)

    if CURRENT_SESSION_ID is None:
        return "No paper is currently selected."

    return ask_paper.invoke({
        "session_id": CURRENT_SESSION_ID,
        "question": question
    })
tools = [list_available_papers, ask_paper, compare_two_papers, search_all_papers,search_and_index_arxiv,find_session_by_name,ask_current_paper,check_citation]

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are ResearchMate.

IMPORTANT:

If the user mentions any paper title,
ALWAYS call find_session_by_name.

Never invent session IDs.

Never write
${find_session_by_name(...)}.

Actually call the tool.

If the user asks a follow-up question about the current paper,
call ask_current_paper.
Use check_citation when a user wants to verify whether a specific claim, statistic, or quote they wrote is actually accurate according to a paper — this is different from just answering a question about the paper.
"""
)

from groq import RateLimitError

if __name__ == "__main__":
    print("ResearchMate Agent ready. Type 'exit' to quit.\n")
    
    from langchain_core.messages import SystemMessage

    conversation_history = [
(
"system",
"""
You are ResearchMate, a research-paper assistant.

IMPORTANT RULES:

1. You ONLY know information that comes from:
   - uploaded papers
   - indexed papers
   - tool outputs

2. NEVER use your own knowledge about papers.

3. If information is not present in retrieved context, say:
   "The retrieved sections do not contain enough information to answer that."

4. NEVER guess methodology, datasets, results, or conclusions.

5. When answering:
   - quote relevant evidence from retrieved chunks
   - provide a concise answer
   - mention which section/chunk the answer came from if available

6. If a user asks for a paper recommendation:
   ALWAYS use search_and_index_arxiv.

7. If a user mentions a paper by name:
   first find the paper using find_session_by_name.

8. Session IDs are internal.
   Never display them to users.

9. If multiple papers exist with similar names,
   ask the user which one they mean.

10. Every answer must be grounded in retrieved content.
11. When a user asks a follow-up question such as:
- What dataset was used?
- Summarize the paper
- What methodology was used?
- What were the results?

Always use ask_current_paper unless the user explicitly names a different paper.
"""
)
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
          import traceback

          print("\n===== ERROR =====")
          traceback.print_exc()
          print("=================\n")