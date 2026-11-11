import argparse
from speech_to_text.speechDetection import(
    start_speech_in_background,
    stop_event as stt_stop_event
)
from test_video import main as video_main

def main():
    """Main run method for the Interview Coach"""
    
    parser = argparse.ArgumentParser(description="Run Interview Coach (video + speech)")
    parser.add_argument(
        "--model",
        type=str,
        default="../models/emotion_cnn.pt",
        help="Path to trained emotion model file (.pt)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default 0)"
    )
    args = parser.parse_args()
    
    print("Starting speech-to-text in background...")
    speech_thread = start_speech_in_background()
    
    try:
        video_main(args.model, args.camera)
    finally:
        print("Stopping speech-to-text...")
        try:
            stt_stop_event.set()
        except Exception:
            pass
        
        try:
            speech_thread.join(timeout=2.0)
        except Exception:
            pass
        
        print("All components stopped. Exiting")
        
if __name__ == "__main__":
    main()
