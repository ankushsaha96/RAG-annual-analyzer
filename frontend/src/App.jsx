import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { AlertCircle, User, Bot, ShieldCheck } from 'lucide-react';
import { ChatInput } from './components/ChatInput';
import './App.css';

const API_URL = 'https://rag-annual-report-analyzer.onrender.com'; // Define backend URL

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;

    // Add user message
    const userMsg = { role: 'user', content: searchQuery, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: searchQuery }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Parse the JSON answer if available, otherwise fallback
      let answerText = data.answer;
      let confidence = null;

      if (data.answer_json && data.answer_json.answer) {
        answerText = data.answer_json.answer;
        if (data.answer_json.confidence) {
          confidence = data.answer_json.confidence;
        }
      } else {
        // Fallback: try to parse manually if data.answer is a stringified JSON
        try {
          const parsed = JSON.parse(data.answer);
          if (parsed.answer) {
            answerText = parsed.answer;
            if (parsed.confidence) {
              confidence = parsed.confidence;
            }
          }
        } catch (e) {
          // not valid JSON, just use as string
        }
      }

      const botMsg = {
        role: 'assistant',
        content: answerText,
        confidence: confidence,
        chunks: data.chunks,
        id: Date.now() + 1
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error("Failed to fetch query results", err);
      const errorMsg = {
        role: 'error',
        content: "Failed to connect to the RAG backend. If using Render, the server might be waking up (can take ~50 seconds). Please try again shortly.",
        id: Date.now() + 1
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header" style={{ marginBottom: messages.length > 0 ? '2rem' : '3rem', transition: 'all 0.3s' }}>
        <h1 style={{ fontSize: messages.length > 0 ? '2rem' : '3rem' }}>RAG Analyzer</h1>
        {messages.length === 0 && <p>Interactive insights from your annual reports.</p>}
      </header>

      <main className="main-content" style={{ paddingBottom: '120px' }}>
        {messages.length === 0 && !isLoading && (
          <div className="glass-panel animate-fade-in" style={{ textAlign: 'center', padding: '4rem 2rem', borderStyle: 'dashed' }}>
            <div style={{ margin: '0 auto', width: '64px', height: '64px', borderRadius: '50%', background: 'var(--bg-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: 'var(--accent-color)' }}>
              <span style={{ fontSize: '1.5rem' }}>✨</span>
            </div>
            <h3 style={{ marginBottom: '0.5rem', fontSize: '1.25rem' }}>Ready to analyze</h3>
            <p style={{ color: 'var(--text-secondary)' }}>Ask your first question about the annual results below.</p>
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
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Analyzer API</span>

                    {msg.confidence && (
                      <div className={`confidence-bubble confidence-${msg.confidence.toLowerCase()}`}>
                        <ShieldCheck size={14} />
                        <span>{msg.confidence} confidence</span>
                      </div>
                    )}
                  </div>

                  <div className="markdown-content">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
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
