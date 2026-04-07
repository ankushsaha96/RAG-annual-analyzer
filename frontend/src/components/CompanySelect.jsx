import React, { useState } from 'react';
import { Building2, Calendar, ArrowRight, Sparkles } from 'lucide-react';
import './components.css';

export function CompanySelect({ onSubmit, isLoading }) {
  const [companyName, setCompanyName] = useState('');
  const [year, setYear] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (companyName.trim() && year && !isLoading) {
      onSubmit(companyName.trim(), parseInt(year));
    }
  };

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 10 }, (_, i) => (currentYear - 1) - i);

  return (
    <div className="select-screen">
      {/* Animated background orbs */}
      <div className="bg-orb bg-orb-1"></div>
      <div className="bg-orb bg-orb-2"></div>
      <div className="bg-orb bg-orb-3"></div>

      <div className="select-card animate-fade-in">
        <div className="select-icon-ring">
          <Sparkles size={32} />
        </div>

        <h1 className="select-title">FinSight AI</h1>
        <p className="select-subtitle">
          Unlock intelligent insights from annual reports.<br />
          Enter the company and financial year to begin.
        </p>

        <form onSubmit={handleSubmit} className="select-form">
          <div className="select-field">
            <label className="select-label" htmlFor="company-name">
              <Building2 size={16} />
              Company Name
            </label>
            <input
              id="company-name"
              type="text"
              className="select-input"
              placeholder="e.g. TCS, Infosys, Reliance"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              disabled={isLoading}
              autoFocus
            />
          </div>

          <div className="select-field">
            <label className="select-label" htmlFor="year-select">
              <Calendar size={16} />
              Financial Year (End Year)
            </label>
            <select
              id="year-select"
              className="select-input select-dropdown"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              disabled={isLoading}
            >
              <option value="">Select year…</option>
              {years.map((y) => (
                <option key={y} value={y}>
                  FY {y - 1}-{String(y).slice(-2)}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="select-button"
            disabled={isLoading || !companyName.trim() || !year}
          >
            <span>Start Analyzing</span>
            <ArrowRight size={18} />
          </button>
        </form>

        <p className="select-hint">
          Reports are fetched from NSE India and indexed on the fly.
        </p>
      </div>
    </div>
  );
}
