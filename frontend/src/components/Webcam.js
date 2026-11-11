import React, { useEffect, useState } from 'react';
import './Webcam.css';

// shows the video and gets stats
function Webcam({ isRunning, onStatsUpdate }) {
  // stores the coaching message like "slow down"
  const [interruption, setInterruption] = useState(null);
  
  // gets stats from backend every 500ms
  useEffect(() => {
    if (!isRunning) return;
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:6767/stats');
        const data = await res.json();
        // sends stats back up to the main app
        onStatsUpdate({
          emotion: data.emotion,
          eyeContact: data.eyeContact,
          wpm: data.wpm,
          nervousness: data.nervousness,
          fillerWords: data.fillerWords,
        });
      } catch (err) {
        console.error('stats error:', err);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [isRunning, onStatsUpdate]);

  // checks for coaching tips from the rl model
  useEffect(() => {
    if (!isRunning) {
      setInterruption(null);
      return;
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:6767/interruption');
        const data = await res.json();
        if (data.active && data.message) {
          setInterruption(data.message);
        } else {
          setInterruption(null);
        }
      } catch (err) {
        // ignore
      }
    }, 500);

    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className="webcam-container">
      {isRunning ? (
        <>
          <img src="http://localhost:6767/video" alt="video feed" />

          {interruption && (
            <div className="interruption-overlay">
              <p>{interruption}</p>
            </div>
          )}
        </>
      ) : (
        <div className="webcam-placeholder">
          <p>camera will appear here</p>
        </div>
      )}
    </div>
  );
}

export default Webcam;
