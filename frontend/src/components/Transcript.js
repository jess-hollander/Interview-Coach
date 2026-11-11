import React, { useEffect, useRef } from 'react';
import './Transcript.css';

// transcript panel, shows what you said
function Transcript({ lines }) {
  const containerRef = useRef(null);

  // scrolls to bottom when new stuff shows up
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="transcript-panel">
      <h3>transcript</h3>
      <div className="transcript-content" ref={containerRef}>
        {lines.length === 0 ? (
          <p className="transcript-empty">your speech will appear here...</p>
        ) : (
          lines.map((line, i) => (
            <div key={i} className="transcript-line">
              <span className="transcript-time">{line.time}</span>
              <span className="transcript-text">{line.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default Transcript;
