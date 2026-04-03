import React from 'react';
import { BookOpen, FileText } from 'lucide-react';
import './components.css';

export function SourceCards({ chunks }) {
  if (!chunks || chunks.length === 0) return null;

  return (
    <div className="sources-container animate-fade-in">
      <div className="sources-header">
        <BookOpen size={18} className="sources-icon" />
        <h3>Retrieved Context</h3>
      </div>
      <div className="sources-grid">
        {chunks.map((chunk, index) => (
          <div key={index} className="source-card">
            <div className="source-card-header">
              <FileText size={14} className="source-doc-icon" />
              <span className="source-page">Page {chunk.page_number}</span>
              <span className="source-score">Relevance: {(chunk.score * 100).toFixed(1)}%</span>
            </div>
            <div className="source-content">
              {chunk.text}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
