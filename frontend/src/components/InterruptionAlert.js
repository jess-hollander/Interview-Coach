import React, { useState, useEffect } from 'react';
import './InterruptionAlert.css';

// shows coaching tips from the rl model
// not actually used rn we show these in the webcam overlay instead
function InterruptionAlert({ isRunning }) {
  const [message, setMessage] = useState(null);

  // gets interruption messages from backend
  useEffect(() => {
    if (!isRunning) {
      setMessage(null);
      return;
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:6767/interruption');
        const data = await res.json();
        
        if (data.active && data.message) {
          setMessage(data.message);
        } else {
          setMessage(null);
        }
      } catch (err) {
        // ignore
      }
    }, 500);

    return () => clearInterval(interval);
  }, [isRunning]);

  if (!message) return null;

  return (
    <div className="interruption-alert">
      <p>{message}</p>
    </div>
  );
}

export default InterruptionAlert;
