import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from pypdf import PdfReader
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from arxiv_search import search_arxiv, download_pdf
import uuid
from agent import supervisor_executor
from groq import RateLimitError
from rag_core import read_pdf_bytes, chunk_text, store_chunks, embed_query, chroma_client
from database import get_db, User, Team, TeamMember, Base, engine
from auth import hash_password, verify_password, create_access_token, get_current_user
from sqlalchemy.orm import Session
from fastapi import Depends
from pydantic import BaseModel, EmailStr
CURRENT_SESSION_ID = None
CURRENT_PAPER_TITLE = None
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
groq_client=Groq(api_key=os.getenv("GROQ_API_KEY"))
sessions={}

@app.get("/sessions")
def list_sessions():
    collections = chroma_client.list_collections()
    
    session_list = []
    for collection in collections:
        session_list.append({
            "session_id": collection.name,
            "chunk_count": collection.count()
        })
    
    return {"sessions": session_list}

def ask_question(collection, question):
    question_embedding=[embed_query(question)]
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
async def upload_paper(
    file: UploadFile = File(...),
    team_id: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()

    MAX_FILE_SIZE = 10 * 1024 * 1024
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

    # If team_id provided, verify user is actually a member
    if team_id:
        membership = db.query(TeamMember).filter(
            TeamMember.team_id == team_id,
            TeamMember.user_id == current_user.id
        ).first()
        if not membership:
            raise HTTPException(status_code=403, detail="You are not a member of this team")

    text = read_pdf_bytes(file_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this PDF")

    chunks = chunk_text(text)

    MAX_CHUNKS = 200
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(status_code=413, detail="Document too long to process. Maximum ~200 chunks supported")

    session_id = str(uuid.uuid4())
    paper_title = file.filename.replace(".pdf", "")

    collection = store_chunks(chunks, session_id, paper_title, "uploaded_file",team_id=team_id,uploaded_by=current_user.email)

    global CURRENT_SESSION_IDs
    global CURRENT_PAPER_TITLE
    CURRENT_SESSION_ID = session_id
    CURRENT_PAPER_TITLE = paper_title

    return {
        "session_id": session_id,
        "paper_title": paper_title,
        "user_id": current_user.id,
        "team_id": team_id,
        "message": f"Paper '{paper_title}' uploaded and processed into {len(chunks)} chunks",
        "chunk_count": len(chunks)
    }
@app.post("/ask")
def ask(session_id: str, question: str):
    try:
        collection = chroma_client.get_collection(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found. Upload the paper first.")
    
    answer, sources = ask_question(collection, question)
    
    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }
@app.post("/search_and_ask")
def search_and_ask(topic:str,question:str):
    results=search_arxiv(topic,max_results=1)
    if not results:
        raise HTTPException(status_code=404, detail="No papers found for this topic")
    
    paper = results[0]
    pdf_bytes=download_pdf(paper["pdf_url"])
    text = read_pdf_bytes(pdf_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the found paper")
    chunks = chunk_text(text)
    session_id = str(uuid.uuid4())
    collection = store_chunks(chunks, session_id)
    sessions[session_id] = collection
    answer, sources = ask_question(collection, question)
    return {
        "paper_found": paper["title"],
        "pdf_url": paper["pdf_url"],
        "question": question,
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }
@app.post("/compare")
def compare_papers(session_ids: list[str], question: str):
    question_embedding=[embed_query(question)]
    all_contexts = []
    all_sources = {}
    for i, session_id in enumerate(session_ids):
        try:
            collection = chroma_client.get_collection(session_id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        results = collection.query(query_embeddings=question_embedding, n_results=3)
        chunks = results["documents"][0]
        label = f"PAPER {chr(65 + i)}"  # PAPER A, PAPER B, PAPER C, etc.
        context = "\n\n".join(chunks)
        all_contexts.append(f"{label}:\n{context}")
        all_sources[label] = chunks
    combined_context = "\n\n---\n\n".join(all_contexts)
    prompt = f"""Compare the following {len(session_ids)} papers based on the question asked.

{combined_context}
Question: {question}
Provide a clear comparison, explicitly referencing what each paper says by its label (PAPER A, PAPER B, etc.)."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "system",
            "content": "You are a research assistant skilled at comparing multiple academic papers. Be specific about which paper each point comes from."
        }, {
            "role": "user",
            "content": prompt
        }]
    )
    return {
        "question": question,
        "papers_compared": len(session_ids),
        "comparison": response.choices[0].message.content,
        "sources": all_sources
    }
import re

def clean_response(text):
    # Remove citation marker artifacts like 【4†L1-L4】
    return re.sub(r'【[^】]*】', '', text).strip()

agent_conversations={}
@app.post("/agent-chat")
def agent_chat(session_id: str, message: str,current_user:User=Depends(get_current_user),db:Session=Depends(get_db)):
    if session_id not in agent_conversations:
        agent_conversations[session_id] = [
            ("system", "You are a research assistant. You ONLY know about papers that have been uploaded or indexed into this system — you have NO knowledge of any other papers, including famous ones from your training data. NEVER invent a paper's title or content. If a user asks to find papers on a topic that isn't already available, use the search_and_index_arxiv tool to find and index a REAL paper before answering. Never substitute a well-known paper name you remember from training. When you find or reference a paper, ALWAYS include its real title and PDF link in your response if available — do not omit them even if you think the user only wants the session_id. NEVER generate fake tool results or pretend you called a tool when you did not. If you don't have specific information (like a PDF link) from an actual previous tool call in this conversation, say so honestly — do not search again or invent a new paper unless the user explicitly asks for a different one,NEVER mention, display, or reference session_ids in your responses to the user — they are internal implementation details. Refer to papers only by their title or topic. Session_ids are for your internal tool use only, never for the user to see.")
        ]
    agent_conversations[session_id].append(("human", message))
    
    try:
        result = supervisor_executor.invoke({"messages": agent_conversations[session_id]})
        final_message = clean_response(result["messages"][-1].content)
        
        # HARD VERIFICATION: extract any session_id mentioned and confirm it actually exists
        
        mentioned_ids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', final_message)
        real_ids = {c.name for c in chroma_client.list_collections()}
        
        fake_ids = [mid for mid in mentioned_ids if mid not in real_ids]
        if fake_ids:
            final_message = "I wasn't able to verify a real paper for that request. Could you try asking again, perhaps with a more specific topic?"
        
        if not final_message or len(final_message.strip()) < 10:
            return {"answer": "I wasn't able to generate a proper answer for that question. Could you try rephrasing it?"}
        
        agent_conversations[session_id].append(("ai", final_message))
        return {"answer": final_message, "session_id": session_id}
        
    except RateLimitError:
        raise HTTPException(status_code=429, detail="We've hit our usage limit for now. Please try again in a few minutes.")
    except Exception as e:
       import traceback
       traceback.print_exc()
       raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/signup")
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        full_name=request.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": user.id})
    return {"token": token, "user_id": user.id, "email": user.email, "full_name": user.full_name}
@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token({"sub": user.id})
    return {"token": token, "user_id": user.id, "email": user.email, "full_name": user.full_name}
@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {"user_id": current_user.id, "email": current_user.email, "full_name": current_user.full_name}

class CreateTeamRequest(BaseModel):
    name:str
class InviteRequest(BaseModel):
    team_id: str
    email: str
@app.post("/teams/create")
def create_team(
    request: CreateTeamRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    print(f"Creating team for user: {current_user.id}")
    
    team = Team(
        name=request.name,
        created_by=current_user.id
    )
    db.add(team)
    db.commit()
    db.refresh(team)
    
    print(f"Team created with ID: {team.id}")
    
    owner_membership = TeamMember(
        team_id=team.id,
        user_id=current_user.id,
        role="owner"
    )
    db.add(owner_membership)
    db.commit()
    
    print(f"Membership added for team: {team.id}")
    
    return {
        "team_id": team.id,
        "team_name": team.name,
        "created_by": current_user.email,
        "message": f"Team '{team.name}' created successfully"
    }
@app.post("/teams/invite")
def invite_to_team(request:InviteRequest,current_user:User=Depends(get_current_user),db:Session=Depends(get_db)):
     # Check the inviter is actually a member of this team
    membership = db.query(TeamMember).filter(
        TeamMember.team_id == request.team_id,
        TeamMember.user_id == current_user.id
    ).first()
    if not membership:
        raise HTTPException(status_code=403, detail="You are not a member of this team")
    #find the user being invited
    invited_user = db.query(User).filter(User.email == request.email).first()
    if not invited_user:
        raise HTTPException(status_code=404, detail="No user found with that email — they need to sign up first")
    # Check they're not already a member
    existing = db.query(TeamMember).filter(
        TeamMember.team_id == request.team_id,
        TeamMember.user_id == invited_user.id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="This person is already a member of the team")
    new_member = TeamMember(
        team_id=request.team_id,
        user_id=invited_user.id,
        role="member"
    )
    db.add(new_member)
    db.commit()
    return {
        "message": f"{invited_user.email} has been added to the team",
        "team_id": request.team_id
    }
@app.get("/teams/my-teams")
def get_my_teams(current_user:User=Depends(get_current_user),db:Session=Depends(get_db)):
    memberships=db.query(TeamMember).filter(TeamMember.user_id==current_user.id).all()
    teams=[]
    for m in memberships:
        team = db.query(Team).filter(Team.id == m.team_id).first()
        member_count = db.query(TeamMember).filter(TeamMember.team_id == team.id).count()
        teams.append({
            "team_id": team.id,
            "team_name": team.name,
            "your_role": m.role,
            "member_count": member_count,
            "created_at": team.created_at
        })
    return {"teams":teams}
@app.get("/teams/{team_id}/papers")
def get_team_papers(team_id:str, current_user:User=Depends(get_current_user),db:Session=Depends(get_db)):
    membership=db.query(TeamMember).filter(TeamMember.team_id==team_id,TeamMember.user_id==current_user.id).first()
    if not membership:
        raise HTTPException(status_code=403,detail="You are not a member of this team")
    collections=chroma_client.list_collections()
    team_papers=[]
    for c in collections:
        collection=chroma_client.get_collection(c.name)
        first_chunk=collection.get(limit=1)
        if first_chunk["metadatas"] and first_chunk["metadatas"][0]:
            meta = first_chunk["metadatas"][0]
            if meta.get("team_id") == team_id:
                team_papers.append({
                    "session_id": c.name,
                    "title": meta.get("title", "Unknown"),
                    "uploaded_by": meta.get("uploaded_by", "Unknown"),
                    "chunk_count": c.count()
                })
    return {"team_id": team_id, "papers": team_papers}