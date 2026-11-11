import React from 'react';
import './Stats.css';

// stats panel, shows all ur metrics in real time
function Stats({ stats, isRunning }) {
  // picks color based on how good or bad the stat is
  // green = good, yellow = alright, red = bad
  const getColor = (value, goodThreshold, badThreshold, inverse = false) => {
    if (inverse) {
      // for stuff where lower is better like nervousness
      if (value < goodThreshold) return '#4ade80';
      if (value < badThreshold) return '#fbbf24';
      return '#ef4444';
    }
    // for stuff where higher is better like eye contact
    if (value > goodThreshold) return '#4ade80';
    if (value > badThreshold) return '#fbbf24';
    return '#ef4444';
  };

  return (
    <div className="stats-panel">
      <h3>live stats</h3>
      
      <div className="stat-item">
        <span className="stat-label">emotion</span>
        <span className="stat-value">{stats.emotion}</span>
      </div>

      <div className="stat-item">
        <span className="stat-label">eye contact</span>
        <span className="stat-value" style={{ color: getColor(stats.eyeContact, 70, 50) }}>
          {stats.eyeContact}%
        </span>
      </div>

      <div className="stat-item">
        <span className="stat-label">speech rate</span>
        <span className="stat-value" style={{ 
          color: stats.wpm > 100 && stats.wpm < 160 ? '#4ade80' : '#fbbf24' 
        }}>
          {stats.wpm} wpm
        </span>
      </div>

      <div className="stat-item">
        <span className="stat-label">nervousness</span>
        <span className="stat-value" style={{ color: getColor(stats.nervousness, 30, 60, true) }}>
          {stats.nervousness}%
        </span>
      </div>

      <div className="stat-item">
        <span className="stat-label">filler words</span>
        <span className="stat-value" style={{ color: getColor(stats.fillerWords, 5, 10, true) }}>
          {stats.fillerWords}%
        </span>
      </div>

      {!isRunning && <p className="stats-hint">start interview to see live stats</p>}
    </div>
  );
}

export default Stats;
