import argparse
import os
from pathlib import Path
import importlib.util
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.rl.q_learning import (
    load_Q,
    save_Q,
    hash_state,
    epsilon_greedy_action,
    compute_reward,
    update_Q,
    NUM_ACTIONS,
    GAMMA,
    EPSILON_START,
    EPSILON_DECAY,
    EPSILON_MIN
)
from models.speech.detection import get_speech_state
from models.eye_tracking.tracker_alt import EyeTracker

FPS_FALLBACK = 60.0 # fallback FPs if the camera doesn't have a valid FPS

# cooldown between RL interruptions (in seconds)
INTERRUPT_COOLDOWN_SECONDS = 0

# thresholds for deciding if a metric is "bad"
NERVOUS_THRESHOLD = 0.5
EYE_CONTACT_THRESHOLD = 0.9
SPEECH_RATE_FAST_THRESHOLD = 220.0
SPEECH_RATE_SLOW_THRESHOLD = 90.0
FILLER_RATE_THRESHOLD = 0.2
SILENCE_LONG_THRESHOLD = 1.5

# order of issue priority when choosing when to add an interruption message
ISSUE_PRIORITY_ORDER = ["no_face", "nervous", "eye_contact", "speech_fast", "speech_slow", "filler", "silence"]
ISSUE_TRIGGER_FRAMES = 5 # how many consecutive "bad" frames to call it an issue
ISSUE_CLEAR_FRAMES = 5 # how many consecutive "good" frames to call it not an issue

def initialize_issues_dict():
    """
    Initialize the dictionary that tracks the status of each issue"
    
    Parameters:
        None
        
    Returns:
        dict: Recorsd each issue that tracks bad frames, good frames, whether it's active, and the interruption message
    """
    return {
        "nervous": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "You seem nervous. Try relaxing and taking a breath."
        },
        "eye_contact": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "Your eyes don't seem centered. Try looking at the camera."
        },
        "speech_fast": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "You're talking a bit fast. Try slowing down a bit."
        },
        "speech_slow": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "You're talking a bit slow. Try speeding up a bit."
        },
        "filler": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "Try to reduce filler words like 'um' and 'like'."
        },
        "silence": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "Try to keep your answer moving. Long silences can feel uncomfortable."
        },
        "no_face": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, "message": "The camera cannot see your face! Ensure you're centered and that we can see you clearly!"
        },
    }

def update_consecutive_issue_counters(issue_state: dict, is_bad: bool) -> None:
    """
    Tracks the total number of consecutive streaks of good/bad frames so we can 
    decide when to turn an issue on or off without flickering
    
    Parameters:
        issue_state (dict): a dictionary representing the state of a single issue
        is_bad (bool): True if the current state's metric is considered "bad"
        
    Returns:
        None
    """
    
    # increment information if we saw another bad frame
    if is_bad:
        issue_state["bad_frames"] += 1
        issue_state["good_frames"] = 0
        
        # if this issue wasn't active yet, but we've crossed the threshold, activate it
        if not issue_state["is_active"] and issue_state["bad_frames"] >= ISSUE_TRIGGER_FRAMES:
            issue_state["is_active"] = True
    else:
        # we saw a good frame so reset the streak
        issue_state["good_frames"] += 1
        issue_state["bad_frames"] = 0
        
        # if the issue is active and we've had enough good frames, set the active false
        if issue_state["is_active"] and issue_state["good_frames"] >= ISSUE_CLEAR_FRAMES:
            issue_state["is_active"] = False

def load_local_module(path: Path, name: str):
    """Load a module from a file path without requiring package imports.
    Returns the module object.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def compute_nervous_score(label: str, confidence: float) -> float:
    """
    Calculate a nervousness score from a given emotion label and confidence score.
    Angry, disgusted, fearful, and sad emotions yield high nervousness.
    Neutral yields medium nervousness.
    Happy and surprised yield low nervousness.
    A higher score is more nervous.
    
    Parameters:
        label (str): one of: 'angry', 'disgusted', 'fearful', 'sad', 'neutral', 'happy', 'surprised'
        confidence (float): a user's calculated confidence score
        
    Returns:
        A float in [0, 1] where a higher number is more nervous and lower to less nervous
    """
    
    negative_labels = {"angry", "disgusted", "fearful", 'sad'}
    
    if label in negative_labels:
        return float(confidence)
    elif label == 'neutral':
        return 0.4 * float(confidence)
    
    return 0.1 * float(confidence)

def main(model_path: str, camera: int = 0):
    # Load CLASS_LABELS
    mod_path = Path(__file__).parent / "emotion_classifier.py"
    if not mod_path.exists():
        print("Could not find emotion_classifier.py next to test_video.py")
        return
    mod = load_local_module(mod_path, "emotion_classifier")
    CLASS_LABELS = getattr(mod, "CLASS_LABELS", None)
    IMG_HEIGHT = getattr(mod, "IMG_HEIGHT", 48)
    IMG_WIDTH = getattr(mod, "IMG_WIDTH", 48)

    # Prefer labels saved during training (models/labels.json) if available.
    labels_path = Path(model_path).parent / "labels.json"
    if labels_path.exists():
        try:
            import json
            with open(labels_path, "r") as fh:
                CLASS_LABELS = json.load(fh)
            print(f"Loaded class labels from {labels_path}")
        except Exception:
            print(f"Failed to load labels from {labels_path}, falling back to module constant.")

    if CLASS_LABELS is None:
        # Fallback to the common ordering used during training if constant is missing
        CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    # Load PyTorch model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}\nPlease provide a trained PyTorch model (.pt).")
        return

    print(f"Loading model from {model_path}...")
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # instantiate the network and load weights (handles common checkpoint formats)
    model = mod.ConvNet(num_classes=len(CLASS_LABELS))
    ckpt = torch.load(model_path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
    if state_dict is None and isinstance(ckpt, dict):
        state_dict = ckpt
    if state_dict is None:
        print("Could not find state_dict in checkpoint. Expecting a PyTorch state dict.")
        return
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    # face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # open webcam
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return
        
    """CONVERT COOLDOWN SECONDS INTO FRAMES"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0 or np.isnan(fps):
        fps = FPS_FALLBACK
    
    # convert cooldown seconds into frames
    interrupt_cooldown_frames = int(INTERRUPT_COOLDOWN_SECONDS * fps)

    """
    RL SETUP => INITIALIZE Q-TABLE
    """
    Q, N = load_Q()
    epsilon = EPSILON_START
    prev_state = None
    prev_state_id = None
    prev_action = None
    
    """INITIALIZE OUR ISSUE DICT; SET OUR CURRENTLY FOCUSED ISSUE TO NONE"""
    issues = initialize_issues_dict()
    current_issue_key = None
    
    """INITIALIZE FRAME-TRACKING INFORMATION"""
    frame_index = 0
    last_interrupt_frame = -interrupt_cooldown_frames
    
    """EYE TRACKING SETUP"""
    eye_tracker = None
    eye_contact_score = 1.0 # neutral/assumed focused
    try:
        eye_tracker = EyeTracker()
    except Exception as e:
        print(f"Eye tracker unavailable, continuing without eye tracking: {e}")
    
    print(f"Camera opened. Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            """INCREMENT CAMERA FRAME COUNT"""
            frame_index += 1
            
            frame_h, frame_w = frame.shape[:2]

            """EYE TRACKING"""
            if eye_tracker is not None:
                eye_result = eye_tracker.process_frame(frame)
                if eye_result is not None:
                    eye_contact_score = float(eye_result["prob_focused"])
                    eye_state_name = eye_result["state"]
                else:
                    eye_state_name = "UNKNOWN"
            else:
                eye_state_name = "DISABLED"

            """RL"""
            nervous_score = 0.0
            
            # pull the smoothed speech metrics from speechDetection
            try:
                speech_rate_wpm, filler_rate, silence_duration = get_speech_state()
            except Exception:
                speech_rate_wpm = 150.0 # fixed for now
                filler_rate = 0.05 # fixed for now
                silence_duration = 0.2 # fixed for now
            
            label = None # emotion label (Emotion classifier)
            confidence = 0.0 # confidence label (Emotion classifier)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            """CHECK IF A FACE IS FOUND IN FRAME"""
            saw_face = len(faces) > 0

            for (x, y, w, h) in faces:
                # Expand the box slightly for better context
                pad_w = int(0.15 * w)
                pad_h = int(0.2 * h)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)

                face = frame[y1:y2, x1:x2]
                # Convert to grayscale and resize to model input
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                face_array = face_resized.astype(np.float32) / 255.0
                face_tensor = torch.from_numpy(face_array).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = F.softmax(logits, dim=1)
                    idx = int(torch.argmax(probs, dim=1).item())
                    conf = float(probs[0, idx].item())

                label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else str(idx)

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                """CALCULATE SCORES FROM CAMERA/AUDIO (UNFINISHED)"""
                nervous_score = compute_nervous_score(label, conf)

            """IF A FACE WAS SEEN, UPDATE OUR ISSUES """
            if saw_face:  
                update_consecutive_issue_counters(issues["nervous"], nervous_score >= NERVOUS_THRESHOLD)
                update_consecutive_issue_counters(issues["eye_contact"], eye_contact_score <= EYE_CONTACT_THRESHOLD)
                update_consecutive_issue_counters(issues["speech_fast"], speech_rate_wpm >= SPEECH_RATE_FAST_THRESHOLD)
                update_consecutive_issue_counters(issues["speech_slow"], speech_rate_wpm <= SPEECH_RATE_SLOW_THRESHOLD)
                update_consecutive_issue_counters(issues["filler"], filler_rate >= FILLER_RATE_THRESHOLD)
                update_consecutive_issue_counters(issues["silence"], silence_duration >= SILENCE_LONG_THRESHOLD)
                update_consecutive_issue_counters(issues["no_face"], False)
            else:
                update_consecutive_issue_counters(issues["no_face"], True)
                update_consecutive_issue_counters(issues["nervous"], False)
                update_consecutive_issue_counters(issues["eye_contact"], False)
                update_consecutive_issue_counters(issues["speech_fast"], False)
                update_consecutive_issue_counters(issues["speech_slow"], False)
                update_consecutive_issue_counters(issues["filler"], False)
                update_consecutive_issue_counters(issues["silence"], False)

            """INITIALIZE CURRENT STATE"""
            curr_state = np.array(
                [
                    float(nervous_score),
                    float(eye_contact_score),
                    float(speech_rate_wpm),
                    float(filler_rate),
                    float(silence_duration)
                ],
                dtype=float
            )
            curr_state_id = hash_state(curr_state)

            """RL ONLINE UPDATE GIVEN A PREVIOUS STATE/ACTION"""
            if prev_state is not None and prev_action is not None and prev_state_id is not None:
                reward = compute_reward(prev_state, curr_state, prev_action)
                update_Q(Q, N, prev_state_id, curr_state_id, prev_action, reward, GAMMA)
                
            """CHOOSE CURRENT ACTION WITH EPSILON-GREEDY"""
            action = epsilon_greedy_action(Q, curr_state_id, num_actions=NUM_ACTIONS, epsilon=epsilon)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            
            """MANAGE INTERRUPTION"""
            applied_action = 0
            can_interrupt = frame_index - last_interrupt_frame >= interrupt_cooldown_frames
            
            # If RL selected "interrupt" and our cooldown has passed, then look
            # at the highest priority issue and set that as our new chosen issue key
            if current_issue_key is None and action == 1 and can_interrupt:
                for key in ISSUE_PRIORITY_ORDER:
                    if issues[key]["is_active"]:
                        current_issue_key = key
                        applied_action = 1
                        last_interrupt_frame = frame_index
                        break
            else:
                # if we already are in the middle of an interuption, keep counting it as applied
                if current_issue_key is not None:
                    applied_action = 1
                  
            """RENDER INTERRUPTION MESSAGE"""
            # if we are in the middle of displaying a message for some issue, either
            # continue displaying it or clear it if the issue is fixed
            if current_issue_key is not None:
                if not issues[current_issue_key]["is_active"]:
                    current_issue_key = None
                else:
                    msg = issues[current_issue_key]["message"]
                
                    # measure text size so we can generate a box behind it
                    (text_w, text_h), text_baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    box_padding = 10
                    center_x = frame_w // 2
                    center_y = frame_h // 2
                    
                    # compute box coordinates
                    x1 = center_x - text_w // 2 - box_padding
                    y1 = center_y - text_h // 2 - box_padding
                    x2 = center_x + text_w // 2 + box_padding
                    y2 = center_y + text_h // 2 + box_padding
                    
                    # clamp the box so it stays on the screen
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_w - 1, x2)
                    y2 = min(frame_h - 1, y2)
                    
                    # draw a background for overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    alpha = 0.5
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
                    # draw white text on the top, centered in the box
                    org = (center_x - text_w // 2, center_y + text_h // 2)
                    
                    # draw the message text in white, centered within the box
                    cv2.putText(
                        frame,
                        msg,
                        org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            
            """HUD INFORMATION"""
            hud_lines = [
                f"WPM: {speech_rate_wpm:.2f}",
                f"Fillers/sec: {filler_rate:.2f}",
                f"Silence Duration: {silence_duration:.1f}s",
                f"Nervous: {nervous_score:.2f}",
                f"Emotion: {label or 'none'} ({confidence:.2f})",
                f"Eye: {eye_state_name} ({eye_contact_score:.2f})",
                f"Issue: {current_issue_key or 'none'}"
            ]
            
            hud_font = cv2.FONT_HERSHEY_SIMPLEX
            hud_scale = 0.5
            hud_thickness = 1
            padding = 5
            
            # compute HUD background size
            max_width = 0
            line_height = 0
            for line in hud_lines:
                (tw, th), _ = cv2.getTextSize(line, hud_font, hud_scale, hud_thickness)
                max_width = max(max_width, tw)
                line_height = max(line_height, th)
            
            hud_height = line_height * len(hud_lines) + padding * (len(hud_lines) + 1)
            hud_width = max_width + 2 * padding
            
            x1 = 10
            y1 = frame_h - hud_height - 10
            x2 = x1 + hud_width
            y2 = y1 + hud_height
            
            # draw HUD background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # render each HUD line
            y_text= y1 + padding + line_height
            for line in hud_lines:
                cv2.putText(
                    frame,
                    line,
                    (x1 + padding, y_text),
                    hud_font,
                    hud_scale,
                    (255, 255, 255),
                    hud_thickness,
                    cv2.LINE_AA,
                )
                y_text += line_height + padding
            
            # update our previous state as the current state we've just gone through;
            # this is what the reward function will see
            prev_state = curr_state
            prev_state_id = curr_state_id
            prev_action = applied_action

            cv2.imshow("Emotion Webcam + RL", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # print a summary of how long each issue was "bad" for during the session
        print("-----SESSION SUMMARY-----")
        
        for key in ISSUE_PRIORITY_ORDER:
            state = issues[key]
            
            # convert the total numer of bad frames into seconds and print
            active_seconds = state["bad_frames"] / fps
            print(f"\t{key}: active for {active_seconds:.2f}s")
        
        # save our learned Q and N tables so that future runs can start from the current policy instead from scratch
        try:
            save_Q(Q, N)
            print(f"Saved Q-table")
        except Exception as e:
            print(f"Could not save Q-table: {e}")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time emotion detection using a trained PyTorch model")
    parser.add_argument("--model", type=str, default="models/emotion_cnn.pt", help="Path to trained model file (.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()
    main(args.model, args.camera)
