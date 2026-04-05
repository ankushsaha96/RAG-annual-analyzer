import React from 'react';
import './components.css';

export function LoadingScreen({ companyName, year }) {
  return (
    <div className="loading-screen">
      {/* Animated background */}
      <div className="bg-orb bg-orb-1"></div>
      <div className="bg-orb bg-orb-2"></div>

      <div className="loading-card-minimal animate-fade-in">
        <div className="shimmer-text-wrapper">
          <h1 className="shimmer-text">Fetching report....</h1>
        </div>
        <p className="loading-company-label">
          {companyName} · FY {year - 1}-{String(year).slice(-2)}
        </p>
      </div>
    </div>
  );
}
