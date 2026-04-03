import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import './components.css';

export function ChatInput({ onSearch, isLoading }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query);
      setQuery('');
    }
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className={`search-input-wrapper ${isLoading ? 'disabled' : ''}`}>
          <Search className="search-icon" size={20} />
          <input
            type="text"
            className="search-input"
            placeholder="Ask about the annual results..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" className="search-button" disabled={isLoading || !query.trim()}>
            {isLoading ? <Loader2 className="spinner" size={20} /> : 'Analyze'}
          </button>
        </div>
      </form>
    </div>
  );
}
