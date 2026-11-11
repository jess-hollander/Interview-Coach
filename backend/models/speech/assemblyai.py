import sounddevice as sd
import websocket
import json
import threading
import time
import os
from urllib.parse import urlencode
from datetime import datetime
import psutil
from collections import deque
import numpy as np
from dotenv import load_dotenv
# Note: This file is our initial attempt at integrating AssemblyAI's real-time streaming API.
# This approach was replaced by our hybrid method using Whisper for better performance and reliability.

# --- Configuration ---
load_dotenv()

CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Audio Configuration
FRAMES_PER_BUFFER = 800  # 50ms of audio (0.05s * 16000Hz)
SAMPLE_RATE = CONNECTION_PARAMS["sample_rate"]
CHANNELS = 1

# Global variables
ws_app = None
audio_queue = deque()
stop_event = threading.Event()
transcript_file = open("./speech_to_text/transcript.txt", "a", encoding="utf-8")
transcript_file.flush()
last_transcript = ""
send_timestamps = deque(maxlen=100)
performance_samples = []
process = psutil.Process()
process.cpu_percent(interval=None)


def audio_callback(indata, frames, time_info, status):
    """Capture audio chunks from microphone and queue for transmission."""
    if status:
        print(f"Audio status: {status}")
    if not stop_event.is_set():
        audio_queue.append(indata.copy().tobytes())


# --- WebSocket Event Handlers ---
def on_open(ws):
    """Start streaming audio from queue to AssemblyAI when WebSocket connects."""
    print("WebSocket connection opened.")
    print(f"Connected to: {API_ENDPOINT}")
    
    def stream_audio():
        print("Starting audio streaming...")
        while not stop_event.is_set():
            try:
                if audio_queue:
                    audio_data = audio_queue.popleft()
                    ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                    send_timestamps.append(time.time())
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy loop
            except Exception as e:
                print(f"Error streaming audio: {e}")
                break
        print("Audio streaming stopped.")
    
    audio_thread = threading.Thread(target=stream_audio, daemon=True)
    audio_thread.start()


def on_message(ws, message):
    """Process transcription messages from AssemblyAI (Begin, Turn, Termination)."""
    try:
        data = json.loads(message)
        msg_type = data.get('type')
        if msg_type == "Begin":
            session_id = data.get('id')
            expires_at = data.get('expires_at')
            print(f"Session began: ID={session_id}, ExpiresAt={datetime.fromtimestamp(expires_at)}")
        elif msg_type == "Turn":
            transcript = data.get('transcript', '')
            formatted = data.get('turn_is_formatted', False)
            global last_transcript
            if formatted and transcript and transcript != last_transcript:
                last_transcript = transcript
                now = time.time()
                latency = now - send_timestamps[0] if send_timestamps else 0
                cpu = process.cpu_percent(interval=None)
                performance_samples.append({
                    "latency": round(latency, 2),
                    "cpu": round(cpu, 2),
                })
                output_block = f"{transcript}\n"
                print(f"\n{output_block}")
                transcript_file.write(output_block)
                transcript_file.flush()
            else:
                print(f"\r{transcript}", end='')
        elif msg_type == "Termination":
            audio_duration = data.get('audio_duration_seconds', 0)
            session_duration = data.get('session_duration_seconds', 0)
            print(f"Session Terminated: Audio Duration={audio_duration}s, Session Duration={session_duration}s")
    except json.JSONDecodeError as e:
        print(f"Error decoding message: {e}")
    except Exception as e:
        print(f"Error handling message: {e}")

def on_error(ws, error):
    """Handle WebSocket errors by signaling shutdown."""
    print(f"WebSocket Error: {error}")
    stop_event.set()

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket closure and cleanup."""
    print(f"WebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
    stop_event.set()


# --- Main Execution ---
def run():
    """Main entry point: establish WebSocket connection and stream microphone audio."""
    global ws_app
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is missing; add it to the project root .env file.")
    
    print("Microphone stream opened successfully.")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    # Create WebSocketApp
    ws_app = websocket.WebSocketApp(
        API_ENDPOINT,
        header={"Authorization": api_key},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    
    # Start audio input stream using sounddevice
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=FRAMES_PER_BUFFER,
        callback=audio_callback
    )
    
    # Run WebSocketApp in a separate thread
    ws_thread = threading.Thread(target=ws_app.run_forever, daemon=True)
    ws_thread.start()
    
    try:
        with stream:
            while ws_thread.is_alive():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Ctrl+C received. Stopping...")
        stop_event.set()
        if ws_app and ws_app.sock and ws_app.sock.connected:
            try:
                terminate_message = {"type": "Terminate"}
                print(f"Sending termination message: {json.dumps(terminate_message)}")
                ws_app.send(json.dumps(terminate_message))
                time.sleep(5)
            except Exception as e:
                print(f"Error sending termination message: {e}")
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        stop_event.set()
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)
    finally:
        if performance_samples and transcript_file:
            avg_latency = sum(p["latency"] for p in performance_samples) / len(performance_samples)
            avg_cpu = sum(p["cpu"] for p in performance_samples) / len(performance_samples)
            final_metrics = {
                "latency": round(avg_latency, 2),
                "cpu": round(avg_cpu, 2),
            }
            transcript_file.write(json.dumps(final_metrics) + "\n")
            transcript_file.flush()
        if transcript_file:
            transcript_file.close()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    run()