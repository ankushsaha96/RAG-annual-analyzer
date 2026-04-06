import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { AlertCircle, User, Bot, ShieldCheck, ArrowLeft } from 'lucide-react';
import { ChatInput } from './components/ChatInput';
import { CompanySelect } from './components/CompanySelect';
import { LoadingScreen } from './components/LoadingScreen';
import './App.css';

const API_URL = 'https://rag-annual-report-analyzer.onrender.com'; // Deployed backend on Render

function App() {
  // State machine: 'select' → 'loading' → 'chat'
  const [screen, setScreen] = useState('select');
  const [companyName, setCompanyName] = useState('');
  const [year, setYear] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // ── Step 1: User submits company + year ────────────────
  const handleCompanySubmit = async (name, yr) => {
    setCompanyName(name);
    setYear(yr);
    setError(null);

    try {
      // Check if embeddings already exist
      const res = await fetch(`${API_URL}/api/check-embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_name: name, year: yr }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();

      if (data.exists) {
        // Embeddings exist → go directly to chat
        setScreen('chat');
      } else {
        // Need to create embeddings → show loading screen
        setScreen('loading');
        await createEmbeddings(name, yr);
      }
    } catch (err) {
      console.error('Check embeddings failed:', err);
      setError('Could not connect to the backend. Make sure the API server is running.');
    }
  };

  // ── Step 2: Create embeddings (while loading screen shows) ──
  const createEmbeddings = async (name, yr) => {
    try {
      const res = await fetch(`${API_URL}/api/create-embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_name: name, year: yr }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      // Success → go to chat
      setScreen('chat');
    } catch (err) {
      console.error('Create embeddings failed:', err);
      setError(err.message || 'Failed to create embeddings. Please try again.');
      setScreen('select');
    }
  };

  // ── Step 3: Query in chat mode ─────────────────────────
  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;

    const userMsg = { role: 'user', content: searchQuery, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          company_name: companyName,
          year: year,
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      let answerText = data.answer;
      let confidence = null;

      if (data.answer_json && data.answer_json.answer) {
        answerText = data.answer_json.answer;
        if (data.answer_json.confidence) {
          confidence = data.answer_json.confidence;
        }
      } else {
        try {
          const parsed = JSON.parse(data.answer);
          if (parsed.answer) {
            answerText = parsed.answer;
            if (parsed.confidence) confidence = parsed.confidence;
          }
        } catch (e) { /* not JSON */ }
      }

      const botMsg = {
        role: 'assistant',
        content: answerText,
        confidence,
        chunks: data.chunks,
        id: Date.now() + 1,
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error("Failed to fetch query results", err);
      const errorMsg = {
        role: 'error',
        content: "Failed to connect to the RAG backend. Please try again.",
        id: Date.now() + 1,
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Back to select screen ─────────────────────────────
  const handleBack = () => {
    setScreen('select');
    setMessages([]);
    setCompanyName('');
    setYear(null);
    setError(null);
  };

  // ── Render ─────────────────────────────────────────────

  // Select screen
  if (screen === 'select') {
    return (
      <>
        {error && (
          <div className="global-error animate-fade-in">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}
        <CompanySelect onSubmit={handleCompanySubmit} isLoading={false} />
      </>
    );
  }

  // Loading screen
  if (screen === 'loading') {
    return <LoadingScreen companyName={companyName} year={year} />;
  }

  // Chat screen
  return (
    <div className="app-container">
      <header className="header" style={{ marginBottom: messages.length > 0 ? '2rem' : '3rem', transition: 'all 0.3s' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
          <button onClick={handleBack} className="back-button" title="Change company">
            <ArrowLeft size={18} />
          </button>
          <h1 style={{ fontSize: messages.length > 0 ? '2rem' : '3rem' }}>RAG Analyzer</h1>
        </div>
        <div className="company-badge animate-fade-in">
          <span className="badge-dot"></span>
          {companyName} · FY {year - 1}-{String(year).slice(-2)}
        </div>
        {messages.length === 0 && <p>Interactive insights from the annual report.</p>}
      </header>

      <main className="main-content" style={{ paddingBottom: '120px' }}>
        {messages.length === 0 && !isLoading && (
          <div className="glass-panel animate-fade-in" style={{ textAlign: 'center', padding: '4rem 2rem', borderStyle: 'dashed' }}>
            <div style={{ margin: '0 auto', width: '64px', height: '64px', borderRadius: '50%', background: 'var(--bg-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: 'var(--accent-color)' }}>
              <span style={{ fontSize: '1.5rem' }}>✨</span>
            </div>
            <h3 style={{ marginBottom: '0.5rem', fontSize: '1.25rem' }}>Ready to analyze</h3>
            <p style={{ color: 'var(--text-secondary)' }}>Ask your first question about {companyName}'s annual results below.</p>
          </div>
        )}

        <div className="chat-history">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-bubble-container ${msg.role === 'user' ? 'user-container' : 'bot-container'}`}>

              {msg.role === 'user' && (
                <div className="chat-bubble user-bubble animate-fade-in">
                  <User size={18} style={{ marginRight: '8px', display: 'inline-block', verticalAlign: 'middle', opacity: 0.8 }} />
                  {msg.content}
                </div>
              )}

              {msg.role === 'assistant' && (
                <div className="chat-bubble bot-bubble animate-fade-in">
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px', gap: '10px' }}>
                    <div className="bot-avatar">
                      <Bot size={20} />
                    </div>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Analyzer</span>

                    {msg.confidence && (
                      <div className={`confidence-bubble confidence-${msg.confidence.toLowerCase()}`}>
                        <ShieldCheck size={14} />
                        <span>{msg.confidence} confidence</span>
                      </div>
                    )}
                  </div>

                  <div className="markdown-content">
                    <ReactMarkdown
                      components={{
                        a: ({node, ...props}) => {
                          if (props.href && props.href.startsWith('#page-')) {
                            const pageNum = parseInt(props.href.replace('#page-', ''));
                            const chunk = msg.chunks?.find(c => parseInt(c.page_number) === pageNum);

                            return (
                              <span className="tooltip-container">
                                <a {...props} className="page-link" onClick={e => e.preventDefault()}>
                                  {props.children}
                                </a>
                                {chunk && (
                                  <span className="tooltip-content">
                                    <div className="tooltip-header">
                                      <span style={{ fontSize: '1.1rem' }}>📄</span> Page {pageNum} Source
                                    </div>
                                    <div style={{ color: 'var(--text-secondary)' }}>
                                      "... {chunk.text.length > 250 ? chunk.text.substring(0, 250) + '...' : chunk.text} ..."
                                    </div>
                                  </span>
                                )}
                              </span>
                            );
                          }
                          return <a {...props} target="_blank" rel="noopener noreferrer" />;
                        }
                      }}
                    >
                      {msg.content?.replace(/\(Page (\d+)\)/gi, '[Page $1](#page-$1)')}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {msg.role === 'error' && (
                <div className="chat-bubble error-bubble animate-fade-in">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#f87171', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    <AlertCircle size={20} />
                    Error
                  </div>
                  {msg.content}
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className={`chat-bubble-container bot-container`}>
              <div className="chat-bubble bot-bubble animate-fade-in" style={{ padding: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px', gap: '10px' }}>
                  <div className="bot-avatar" style={{ animation: 'pulse-glow 1.5s infinite' }}>
                    <Bot size={20} />
                  </div>
                  <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Analyzing context...</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
                  <div style={{ height: '12px', width: '80%', background: 'var(--bg-tertiary)', borderRadius: '6px', animation: 'pulse-glow 1.5s infinite' }}></div>
                  <div style={{ height: '12px', width: '100%', background: 'var(--bg-tertiary)', borderRadius: '6px', animation: 'pulse-glow 1.5s infinite', animationDelay: '0.2s' }}></div>
                  <div style={{ height: '12px', width: '60%', background: 'var(--bg-tertiary)', borderRadius: '6px', animation: 'pulse-glow 1.5s infinite', animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <div style={{ position: 'fixed', bottom: 0, left: 0, right: 0, pointerEvents: 'none', height: '150px', background: 'linear-gradient(to top, var(--bg-primary) 30%, transparent)', zIndex: 50 }}></div>

      <div style={{ position: 'relative', zIndex: 100 }}>
        <ChatInput onSearch={handleSearch} isLoading={isLoading} />
      </div>
    </div>
  );
}

export default App;
