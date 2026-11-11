import React, { useState, useEffect } from 'react';
import Webcam from './components/Webcam';
import Stats from './components/Stats';
import Transcript from './components/Transcript';
import FinalScore from './components/FinalScore';
import './App.css';

// main app component, controls the whole interview thing
function App() {
  // keeps track of if interview is running or not
  const [isRunning, setIsRunning] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  // all the stats we get from the backend
  const [stats, setStats] = useState({
    emotion: 'neutral',
    eyeContact: 85,
    wpm: 120,
    nervousness: 20,
    fillerWords: 5,
  });
  const [transcript, setTranscript] = useState([]);

  // timer that goes up every second
  useEffect(() => {
    if (!isRunning) return;
    
    const interval = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // turns seconds into mins:secs format
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // starts the interview, tells backend to turn on camera and mic
  const handleStart = async () => {
    try {
      await fetch('http://localhost:6767/start', { method: 'POST' });
      setIsRunning(true);
      setShowResults(false);
      setTranscript([]);
      setElapsedTime(0);
    } catch (err) {
      console.error('start error:', err);
      alert('failed to start - is backend running?');
    }
  };

  // stops the interview and shows your score
  const handleStop = async () => {
    try {
      await fetch('http://localhost:6767/stop', { method: 'POST' });
    } catch (err) {
      console.error('stop error:', err);
    }
    setIsRunning(false);
    setShowResults(true);
  };

  // grabs transcript from backend every second
  useEffect(() => {
    if (!isRunning) return;
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:6767/transcript');
        const data = await res.json();
        if (data.lines && data.lines.length > 0) {
          setTranscript(data.lines.map((text, i) => ({ time: '', text })));
        }
      } catch (err) {
        // ignore
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // shows the results page when ur done
  if (showResults) {
    return <FinalScore stats={stats} elapsedTime={elapsedTime} onRestart={() => setShowResults(false)} />;
  }

  // main layout, webcam on left and stats on right
  return (
    <div className="app">
      <header>
        <h1>interview coach</h1>
        <p>practice your interview skills</p>
      </header>

      <main>
        <div className="left-panel">
          <Webcam isRunning={isRunning} onStatsUpdate={setStats} />
          
          <div className="controls">
            {!isRunning ? (
              <button onClick={handleStart} className="btn-start">
                start interview
              </button>
            ) : (
              <>
                <div className="timer">{formatTime(elapsedTime)}</div>
                <button onClick={handleStop} className="btn-stop">
                  end & get score
                </button>
              </>
            )}
          </div>
        </div>

        <div className="right-panel">
          <Stats stats={stats} isRunning={isRunning} />
          <Transcript lines={transcript} />
        </div>
      </main>
    </div>
  );
}

export default App;
