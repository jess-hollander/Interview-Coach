import React from 'react';
import './FinalScore.css';

// final score page shows how you did after the interview
function FinalScore({ stats, elapsedTime, onRestart }) {
  // calc scores from the raw stats
  const eyeScore = stats.eyeContact;
  const speechScore =stats.wpm > 100 && stats.wpm < 160 ? 90 : 60; // 100-160 wpm is good
  const nervousnessScore= 100 - stats.nervousness; // flip it so higher is better
  const fillerScore = 100 - (stats.fillerWords * 5); // more fillers means lower score

  // overall is just the average of everything
  const overall = Math.round((eyeScore + speechScore + nervousnessScore + fillerScore) / 4);
  // turns number score into letter grade
  const getGrade = (score) => {
    if (score>= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
  };

  const getScoreColor = (score) => {
    if (score >= 70) return '#4ade80';
    if (score >= 50) return '#fbbf24';
    return '#ef4444';
  };
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="final-score">
      <h1>interview complete</h1>
      
      <div className="interview-time">
        duration: {formatTime(elapsedTime || 0)}
      </div>
      
      <div className="score-circle" style={{ borderColor: getScoreColor(overall) }}>
        <span className="score-number" style={{ color: getScoreColor(overall) }}>{overall}</span>
        <span className="score-grade">grade: {getGrade(overall)}</span>
      </div>

      <div className="score-breakdown">
        <h3>breakdown</h3>
        <div className="score-item">
          <span>eye contact</span>
          <span style={{ color: getScoreColor(eyeScore) }}>{eyeScore}%</span>
        </div>
        <div className="score-item">
          <span>speech pace</span>
          <span style={{ color: getScoreColor(speechScore) }}>{speechScore}%</span>
        </div>
        <div className="score-item">
          <span>composure</span>
          <span style={{ color: getScoreColor(nervousnessScore) }}>{nervousnessScore}%</span>
        </div>
        <div className="score-item">
          <span>clarity</span>
          <span style= {{ color: getScoreColor(Math.max(0, fillerScore)) }}>{Math.max(0, fillerScore)}%</span>
        </div>
      </div>
      {/* tips based on how you did */}
      <div className="tips">
        <h3>tips</h3>
        {eyeScore < 70 && <p>- try to look at the camera more</p>}
        {stats.wpm > 160 && <p>- slow down a bit when speaking</p>}
        {stats.wpm < 100 && <p>- try to speak a bit faster</p>}
        {stats.nervousness > 40 && <p>- take deep breaths to stay calm</p>}
        {stats.fillerWords > 10 && <p>- pause instead of saying "um"</p>}
        {overall >= 80 && <p>- great job, keep practicing</p>}
      </div>
      <button onClick={onRestart} className="btn-restart">
        try again!
      </button>
    </div>
  );
}

export default FinalScore;
