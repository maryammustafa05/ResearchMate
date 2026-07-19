import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles, ArrowLeft, Send, FileText, Loader2,
  AlertCircle, X, Plus
} from 'lucide-react';
import './AppPage.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

function getAuthHeaders() {
  const token = sessionStorage.getItem('token');
  return { 'Authorization': `Bearer ${token}` };
}

function getSessionId() {
  let id = sessionStorage.getItem('chat_session_id');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('chat_session_id', id);
  }
  return id;
}

export default function AppPage() {
  const [papers, setPapers] = useState([]);
  const [messages, setMessages] = useState([
    { role: 'agent', text: "Upload a paper, or just tell me a topic and I'll find one for you." },
  ]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  // Team state — inside component
  const [teams, setTeams] = useState([]);
  const [currentTeam, setCurrentTeam] = useState(null);
  const [showTeamPanel, setShowTeamPanel] = useState(false);
  const [teamName, setTeamName] = useState('');
  const [inviteEmail, setInviteEmail] = useState('');
  const [teamMessage, setTeamMessage] = useState(null);

  const userEmail = sessionStorage.getItem('user_email');
  const userName = sessionStorage.getItem('user_name');
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const sessionId = useRef(getSessionId());

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  useEffect(() => {
    loadTeams();
  }, []);

  async function loadTeams() {
    try {
      const res = await fetch(`${API_BASE}/teams/my-teams`, {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setTeams(data.teams);
        if (data.teams.length > 0) setCurrentTeam(data.teams[0]);
      }
    } catch (err) {
      console.error('Could not load teams');
    }
  }

  async function createTeam() {
    if (!teamName.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/teams/create`, {
        method: 'POST',
        headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: teamName }),
      });
      const data = await res.json();
      if (res.ok) {
        setTeamMessage(`Team "${data.team_name}" created!`);
        setTeamName('');
        loadTeams();
      } else {
        setTeamMessage(data.detail);
      }
    } catch (err) {
      setTeamMessage('Something went wrong');
    }
  }

  async function inviteMember() {
    if (!inviteEmail.trim() || !currentTeam) return;
    try {
      const res = await fetch(`${API_BASE}/teams/invite`, {
        method: 'POST',
        headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ team_id: currentTeam.team_id, email: inviteEmail }),
      });
      const data = await res.json();
      if (res.ok) {
        setTeamMessage(`${inviteEmail} added to the team!`);
        setInviteEmail('');
      } else {
        setTeamMessage(data.detail);
      }
    } catch (err) {
      setTeamMessage('Something went wrong');
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.endsWith('.pdf')) {
      setError('Only PDF files are supported.');
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData,
      });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();

      setPapers((prev) => [
        ...prev,
        { sessionId: data.session_id, name: file.name, chunks: data.chunk_count },
      ]);
      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          text: `Indexed "${file.name}" into ${data.chunk_count} chunks. Ask me anything about it - try "what's the main contribution?"`,
        },
      ]);
    } catch (err) {
      setError('Could not upload that file. Check your backend is running.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || isThinking) return;

    setMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setIsThinking(true);
    setError(null);

    const paperContext = papers.length > 0
      ? `\n\n[Context: The user has these papers available - ${papers.map(p => `"${p.name}" (session_id: ${p.sessionId})`).join(', ')}. If they refer to a paper by name, use its session_id automatically without asking them for it.]`
      : '';

    try {
      const params = new URLSearchParams({
        session_id: sessionId.current,
        message: text + paperContext,
      });
      const res = await fetch(`${API_BASE}/agent-chat?${params}`, {
        method: 'POST',
        headers: getAuthHeaders(),
      });

      if (res.status === 429) {
        setMessages((prev) => [
          ...prev,
          { role: 'agent', text: "We've hit our usage limit for now. Please try again in a few minutes.", isError: true },
        ]);
        return;
      }
      if (!res.ok) throw new Error('Request failed');

      const data = await res.json();
      setMessages((prev) => [...prev, { role: 'agent', text: data.answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'agent', text: 'Something went wrong reaching the assistant. Please try again.', isError: true },
      ]);
    } finally {
      setIsThinking(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-top">
          <Link to="/" className="sidebar-brand">
            <ArrowLeft size={15} />
            <span>ResearchMate</span>
          </Link>
        </div>

        <div className="user-info">
          <span className="user-name">{userName}</span>
          <span className="user-email">{userEmail}</span>
        </div>

        <div className="team-section">
          <div className="team-header" onClick={() => setShowTeamPanel(!showTeamPanel)}>
            <span className="sidebar-section-label">Team</span>
            <span className="team-toggle">{showTeamPanel ? '▲' : '▼'}</span>
          </div>

          {currentTeam && !showTeamPanel && (
            <div className="current-team-name">{currentTeam.team_name}</div>
          )}

          {showTeamPanel && (
            <div className="team-panel">
              {teams.length === 0 ? (
                <>
                  <p className="team-empty">You're not in a team yet.</p>
                  <input
                    className="team-input"
                    placeholder="Team name"
                    value={teamName}
                    onChange={(e) => setTeamName(e.target.value)}
                  />
                  <button className="team-btn" onClick={createTeam}>
                    Create team
                  </button>
                </>
              ) : (
                <>
                  <div className="team-list">
                    {teams.map(t => (
                      <div
                        key={t.team_id}
                        className={`team-item ${currentTeam?.team_id === t.team_id ? 'team-item-active' : ''}`}
                        onClick={() => setCurrentTeam(t)}
                      >
                        {t.team_name}
                        <span className="team-role">{t.your_role}</span>
                      </div>
                    ))}
                  </div>

                  {currentTeam && (
                    <div className="invite-section">
                      <p className="sidebar-section-label" style={{marginTop: '12px'}}>
                        Invite teammate
                      </p>
                      <input
                        className="team-input"
                        placeholder="their@email.com"
                        value={inviteEmail}
                        onChange={(e) => setInviteEmail(e.target.value)}
                      />
                      <button className="team-btn" onClick={inviteMember}>
                        Send invite
                      </button>
                    </div>
                  )}
                </>
              )}

              {teamMessage && (
                <p className="team-message">{teamMessage}</p>
              )}
            </div>
          )}
        </div>

        <button
          className="upload-zone"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
        >
          {isUploading ? <Loader2 size={17} className="spin" /> : <Plus size={17} />}
          <span>{isUploading ? 'Indexing...' : 'Upload a paper'}</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />

        <div className="sidebar-section-label">Papers in this session</div>
        <div className="paper-list">
          {papers.length === 0 && (
            <p className="paper-list-empty">Nothing uploaded yet.</p>
          )}
          {papers.map((p) => (
            <div key={p.sessionId} className="paper-item">
              <FileText size={15} />
              <div className="paper-item-meta">
                <span className="paper-item-name">{p.name}</span>
                <span className="paper-item-chunks">{p.chunks} chunks</span>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-hint">
          <Sparkles size={13} />
          Don't have a PDF? Just ask me to find one on any topic.
        </div>
      </aside>

      <main className="chat-main">
        <div className="chat-scroll">
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
          {isThinking && <ThinkingBubble />}
          <div ref={chatEndRef} />
        </div>

        {error && (
          <div className="inline-error">
            <AlertCircle size={15} />
            {error}
            <button onClick={() => setError(null)}><X size={14} /></button>
          </div>
        )}

        <div className="composer">
          <input
            className="composer-input"
            placeholder="Ask about a paper, or paste a topic to search..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="composer-send"
            onClick={sendMessage}
            disabled={!input.trim() || isThinking}
            aria-label="Send message"
          >
            <Send size={16} />
          </button>
        </div>
      </main>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  return (
    <div className={`msg-row ${isUser ? 'msg-row-user' : ''}`}>
      {!isUser && (
        <div className="msg-avatar">
          <Sparkles size={14} />
        </div>
      )}
      <div className={`msg-bubble ${isUser ? 'msg-bubble-user' : ''} ${message.isError ? 'msg-bubble-error' : ''}`}>
        {message.text}
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="msg-row">
      <div className="msg-avatar"><Sparkles size={14} /></div>
      <div className="msg-bubble msg-bubble-thinking">
        <span className="think-dot" />
        <span className="think-dot" />
        <span className="think-dot" />
      </div>
    </div>
  );
}