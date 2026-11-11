import sounddevice as sd
import whisper
import requests
import numpy as np
import time
from datetime import datetime
import threading
import queue
import io
import wave
import os
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

load_dotenv()

"""SMOOTHING AND WPM CONSTRAINTS"""
WPM_SMOOTH_WINDOW = 20
MIN_WPM_SAMPLES = 5
MIN_PLAUSIBLE_WPM = 0.0
MAX_PLAUSIBLE_WPM = 260.0

wpm_history = deque(maxlen=WPM_SMOOTH_WINDOW)
speech_rate_wpm = 0.0
filler_rate = 0.0
last_speech_time = time.time()
_speech_lock = threading.Lock()

# global session totals
total_words = 0
total_fillers = 0
total_speaking_time = 0.0
session_start_time = None


def compute_raw_speech_state():
    """Calculate current speech metrics from accumulated session data (thread-safe)."""
    global total_words, total_fillers, total_speaking_time, session_start_time
    with _speech_lock:
        silence = max(0.0, time.time() - last_speech_time)
        if total_words > 0:
            global_filler_rate = total_fillers / total_words
        else:
            global_filler_rate = 0.0
        
        # calculate average wpm over session
        if total_speaking_time > 0:
            avg_wpm = (total_words / total_speaking_time) * 60.0
        else:
            avg_wpm = 0.0
        
        if silence > 4.0:
            return 0.0, global_filler_rate, silence, avg_wpm
        return speech_rate_wpm, global_filler_rate, silence, avg_wpm


def get_speech_state():
    """Get smoothed speech rate with outlier rejection and median filtering."""
    global wpm_history
    raw_wpm, filler_rate_val, silence_duration, avg_wpm = compute_raw_speech_state()
    
    if silence_duration > 4.0:
        return avg_wpm, filler_rate_val, silence_duration
    
    raw_wpm = float(np.clip(raw_wpm, MIN_PLAUSIBLE_WPM, MAX_PLAUSIBLE_WPM))
    wpm_history.append(raw_wpm)
    if len(wpm_history) >= MIN_WPM_SAMPLES:
        smoothed_wpm = float(np.median(wpm_history))
    else:
        smoothed_wpm = raw_wpm
    return smoothed_wpm, filler_rate_val, silence_duration

# Configuration for AssemblyAI API and audio processing
SAMPLE_RATE = 16000
CHANNELS = 1
FRAMES_PER_BUFFER = 512
CHUNK_DURATION = 3.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

def assemblyai_headers():
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is missing; add it to the project root .env file.")
    return {
        "authorization": api_key,
        "content-type": "application/json"
    }

# Global execution state
stop_event = threading.Event()
audio_queue = queue.Queue()
transcript_file = None

FILLER_WORDS = {
    'um', 'uh', 'hmm', 'mm', 'mhm', 'uh-huh', 'mm-hmm',
    'er', 'ah', 'like', 'you know', 'i mean', 'sort of',
    'kind of', 'basically', 'actually', 'literally', 'yeah',
    'so', 'well', 'okay', 'right', 'gonna', 'wanna', 'gotta'
}


def extract_fillers_from_whisper(text):
    words = text.lower().split()
    fillers = []
    prev_filler = None
    consecutive_count = 0
    for word in words:
        word_clean = word.strip('.,!?;:')
        if word_clean in FILLER_WORDS:
            if word_clean == prev_filler:
                consecutive_count += 1
                if consecutive_count <= 3:
                    fillers.append(word_clean)
            else:
                consecutive_count = 1
                fillers.append(word_clean)
            prev_filler = word_clean
        else:
            prev_filler = None
            consecutive_count = 0
    if len(fillers) > 10:
        return []
    return fillers


def merge_transcripts(assembly_text, whisper_fillers):
    """Merge AssemblyAI transcript with Whisper-detected fillers, displayed in yellow."""
    if not whisper_fillers:
        return assembly_text, 0
    words = assembly_text.split()
    result = []
    filler_idx = 0
    for i, word in enumerate(words):
        if filler_idx < len(whisper_fillers) and (
            i == 0 or i == len(words) - 1 or
            word.endswith(',') or
            word.lower() in ['and', 'but', 'so', 'then', 'now']
        ):
            result.append(f"\033[93m[{whisper_fillers[filler_idx]}]\033[0m")
            filler_idx += 1
        result.append(word)
    while filler_idx < len(whisper_fillers):
        result.append(f"\033[93m[{whisper_fillers[filler_idx]}]\033[0m")
        filler_idx += 1
    return ' '.join(result), len(whisper_fillers)

def upload_to_assembly(audio_data):
    """Upload audio to AssemblyAI's temporary storage for transcription."""
    headers = assemblyai_headers()
    try:
        response = requests.post(
            UPLOAD_URL,
            headers={"authorization": headers["authorization"]},
            data=audio_data
        )
        response.raise_for_status()
        return response.json()['upload_url']
    except Exception as e:
        print(f"  ✗ AssemblyAI upload error: {e}")
        return None


def transcribe_with_assembly(audio_url):
    """Submit audio URL for transcription and poll for results."""
    headers = assemblyai_headers()
    try:
        transcript_request = {"audio_url": audio_url, "speech_model": "best"}
        response = requests.post(TRANSCRIPT_URL, json=transcript_request, headers=headers)
        response.raise_for_status()
        transcript_id = response.json()['id']
        polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
        while True:
            response = requests.get(polling_url, headers=headers)
            response.raise_for_status()
            result = response.json()
            status = result['status']
            if status == 'completed':
                return result.get('text', '')
            elif status == 'error':
                print(f"  ✗ AssemblyAI error: {result.get('error')}")
                return None
            time.sleep(0.5)
    except Exception as e:
        print(f"  ✗ AssemblyAI transcription error: {e}")
        return None


def dual_transcribe_worker(whisper_model):
    """Process audio chunks with both AssemblyAI and Whisper in parallel, merge results."""
    global transcript_file, speech_rate_wpm, filler_rate, last_speech_time
    
    print("\n" + "="*60)
    print("HYBRID TRANSCRIPTION ACTIVE")
    print("AssemblyAI: Accurate speech-to-text")
    print("Whisper: Filler word detection")
    print("="*60 + "\n")
    
    filler_stats = {"total": 0, "chunks": 0}
    audio_buffer = []
    chunk_count = 0
    
    while not stop_event.is_set():
        try:
            while len(audio_buffer) < CHUNK_SIZE and not stop_event.is_set():
                try:
                    data = audio_queue.get(timeout=0.1)
                    audio_buffer.extend(data)
                except queue.Empty:
                    continue
            
            if stop_event.is_set():
                break
            
            chunk_count += 1
            print(f"\n[Chunk {chunk_count}] Processing with both engines...", flush=True)
            
            chunk = np.array(audio_buffer[:CHUNK_SIZE], dtype=np.int16)
            audio_buffer = audio_buffer[CHUNK_SIZE:]
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(chunk.tobytes())
            wav_data = wav_buffer.getvalue()
            
            audio_float = chunk.astype(np.float32) / 32768.0
            
            assembly_text = None
            whisper_fillers = []
            
            def run_assembly():
                nonlocal assembly_text
                upload_url = upload_to_assembly(wav_data)
                if upload_url:
                    assembly_text = transcribe_with_assembly(upload_url)
            
            def run_whisper():
                nonlocal whisper_fillers
                try:
                    result = whisper_model.transcribe(
                        audio_float, language='en',
                        initial_prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Um, uh, er, ah, so, yeah, well.",
                        fp16=False, temperature=0.0,
                        no_speech_threshold=0.6, condition_on_previous_text=False
                    )
                    whisper_text = result['text'].strip()
                    print(f"  Whisper raw: {whisper_text[:100]}{'...' if len(whisper_text) > 100 else ''}")
                    whisper_fillers = extract_fillers_from_whisper(whisper_text)
                except Exception as e:
                    print(f"  ✗ Whisper error: {e}")
            
            assembly_thread = threading.Thread(target=run_assembly)
            whisper_thread = threading.Thread(target=run_whisper)
            assembly_thread.start()
            whisper_thread.start()
            assembly_thread.join()
            whisper_thread.join()
            
            if assembly_text:
                merged_text, filler_count = merge_transcripts(assembly_text, whisper_fillers)
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n[{timestamp}] AssemblyAI: {assembly_text}")
                if whisper_fillers:
                    print(f"  Whisper detected fillers: {', '.join(whisper_fillers)}")
                print(f"  Combined: {merged_text}")
                
                if filler_count > 0:
                    filler_stats["total"] += filler_count
                    filler_stats["chunks"] += 1
                    print(f"  → {filler_count} filler(s) | Total: {filler_stats['total']}")
                
                now = time.time()
                words = assembly_text.split()
                num_words = len(words)
                filler_count_effective = filler_count if filler_count > 0 else len([w for w in words if w.lower() in FILLER_WORDS])
                
                if num_words > 0:
                    wpm_new = (num_words / CHUNK_DURATION) * 60.0
                else:
                    wpm_new = speech_rate_wpm
                
                with _speech_lock:
                    global total_words, total_fillers, total_speaking_time
                    total_words += num_words
                    total_fillers += filler_count_effective
                    if num_words > 0:
                        total_speaking_time += CHUNK_DURATION
                    speech_rate_wpm = wpm_new
                    last_speech_time = now
                
                if transcript_file:
                    transcript_file.write(f"[{timestamp}]\n")
                    transcript_file.write(f"  AssemblyAI: {assembly_text}\n")
                    if whisper_fillers:
                        transcript_file.write(f"  Fillers: {', '.join(whisper_fillers)}\n")
                    transcript_file.flush()
            else:
                print(f"  ✗ No transcription available")
                
        except Exception as e:
            if not stop_event.is_set():
                print(f"\nError in worker: {e}")
            time.sleep(0.1)
    
    if filler_stats["chunks"] > 0:
        avg_fillers = filler_stats["total"] / filler_stats["chunks"]
        print(f"\n{'='*60}")
        print(f"Session Summary:")
        print(f"  Total filler words detected: {filler_stats['total']}")
        print(f"  Segments with fillers: {filler_stats['chunks']}")
        print(f"  Average per segment: {avg_fillers:.2f}")
        print(f"{'='*60}\n")


def audio_callback(indata, frames, time_info, status):
    """Capture audio blocks from microphone and queue for processing."""
    if status:
        print(f"Audio status: {status}")
    if not stop_event.is_set():
        # Convert to int16 list and add to queue
        audio_np = (indata[:, 0] * 32768).astype(np.int16)
        audio_queue.put(audio_np.tolist())


def run():
    """Main execution: initialize Whisper model, start audio stream, and run worker thread."""
    global transcript_file, last_speech_time, speech_rate_wpm, filler_rate, total_words, total_fillers, total_speaking_time
    
    stop_event.clear()
    with _speech_lock:
        last_speech_time = time.time()
        speech_rate_wpm = 0.0
        filler_rate = 0.0
        total_words = 0
        total_fillers = 0
        total_speaking_time = 0.0
        wpm_history.clear()
    
    print("Initializing HYBRID transcription system...")
    print("Loading Whisper model for filler detection...\n")
    
    whisper_model = whisper.load_model("tiny")
    
    print("✓ Whisper loaded (tiny model)")
    print("✓ AssemblyAI ready (best model)")
    print("✓ Hybrid system initialized\n")
    
    transcript_dir = Path(__file__).resolve().parent
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = open(transcript_dir / "transcript.txt", "w", encoding="utf-8")
    transcript_file.flush()
    
    print("✓ Microphone connected")
    print("✓ Starting hybrid transcription...\n")
    print("Speak now - using both engines simultaneously")
    print("Press Ctrl+C to stop\n")
    
    # Start dual transcription worker
    worker_thread = threading.Thread(target=dual_transcribe_worker, args=(whisper_model,), daemon=True)
    worker_thread.start()
    
    try:
        # Use sounddevice InputStream with context manager
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=FRAMES_PER_BUFFER,
            callback=audio_callback
        ):
            while not stop_event.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping transcription...")
        stop_event.set()
    except Exception as e:
        print(f"Error with audio stream: {e}")
        stop_event.set()
    finally:
        worker_thread.join(timeout=5)
        if transcript_file:
            transcript_file.write(f"\nSession ended: {datetime.now()}\n")
            transcript_file.close()
        print("Cleanup complete. Goodbye!")


def start_speech_in_background():
    """Launch the hybrid transcription system as a background daemon thread."""
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    run()