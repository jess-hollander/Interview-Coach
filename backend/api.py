# flask backend api for the interview coach app
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import threading
import time
import importlib.util
import sys

# add parent dir to path so we can import our models
sys.path.insert(0, str(Path(__file__).parent))

# import our custom modules for speech, eye tracking, and rl
from models.speech.detection import (
    start_speech_in_background,
    stop_event as stt_stop_event,
    get_speech_state
)
from models.eye_tracking.tracker import EyeTracker
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

# set up flask app with cors so frontend can talk to it
app = Flask(__name__)
CORS(app)

# thresholds for when we consider something a problem
nerv_thresh=0.25
eye_thresh= 0.5
fast_talk_thresh =220.0
slow_talk_thresh = 60.0
filler_thresh = 0.2
silence_thresh = 5.0

# which issues we check first, no face is most important
issue_order =["no_face", "nervous", "eye_contact", "speech_fast", "speech_slow", "filler", "silence"]
trigger_frames = 5
clear_frames = 5

# global state variables
cam= None
running= False
cam_idx =0
eye_ok= False
eye_err =None
model_ok= False
# stores the latest stats that get sent to frontend
stats = {
    "emotion": "neutral",
    "confidence": 0.0,
    "eyeContact": 75,
    "wpm": 120,
    "nervousness": 0,
    "fillerWords": 0,
    "silence": 0.0
}
transcript_lines = []
model = None
face_cascade = None
eye_tracker = None
device = None
CLASS_LABELS = None
speech_thrd = None

# rl stuff for deciding when to show interruptions
Qtable = None
Ntable = None
eps = EPSILON_START
prev_st = None
prev_st_id = None
prev_act = None
problems = None
curr_issue = None
frame_cnt = 0
last_intrpt = 0
interrupt_msg = {"active": False, "type": None, "message": None}

# for thread safe frame sharing between capture and processing
last_frame = None
fr_lock=threading.Lock()
proc_thrd =None

# helper to dynamically load a python module from path
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# calculates how nervous someone looks based on emotion
def compute_nervous_score(label, confidence):
    negative = {"angry", "disgusted", "fearful", "sad"}
    if label in negative:
        return float(confidence)
    elif label == "neutral":
        return 0.3* float(confidence)
    elif label == "happy":
        return 0.05*float(confidence)
    elif label == "surprised":
        return 0.4 * float(confidence)
    return 0.1 * float(confidence)


# sets up dict to track all the different issues we watch for
def initialize_issues_dict():
    return {
        "nervous": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, 
            "message": "You seem nervous. Try relaxing and taking a breath."
        },
        "eye_contact": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, 
            "message": "Your eyes don't seem centered. Try looking at the camera."
        },
        "speech_fast": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, 
            "message": "You're talking a bit fast. Try slowing down a bit."
        },
        "speech_slow": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, 
            "message": "You're talking a bit slow. Try speeding up a bit."
        },
        "filler": {
            "bad_frames": 0, "good_frames": 0, "is_active": False, 
            "message": "Try to reduce filler words like 'um' and 'like'."
        },
        "silence": {
            "bad_frames": 0, "good_frames":0, "is_active": False, 
            "message": "Try to keep your answer moving. Long silences can feel uncomfortable."
        },
        "no_face": {
            "bad_frames":0, "good_frames":0, "is_active": False, 
            "message": "The camera cannot see your face! Ensure you're centered."
        },
    }


# tracks consecutive good/bad frames to avoid flickering
def update_consecutive_issue_counters(issue_state, is_bad):
    if is_bad:
        issue_state["bad_frames"]+=1
        issue_state["good_frames"]=0
        if not issue_state["is_active"] and issue_state["bad_frames"] >= trigger_frames:
            issue_state["is_active"] = True
    else:
        issue_state["good_frames"] += 1
        issue_state["bad_frames"] = 0
        if issue_state["is_active"] and issue_state["good_frames"] >= clear_frames:
            issue_state["is_active"] = False


# loads the emotion cnn model and sets up eye tracker
def init_model():
    global model, face_cascade, eye_tracker, device, CLASS_LABELS
    global eye_ok, eye_err, model_ok
    
    project_root = Path(__file__).parent.parent
    model_path = project_root / "weights/emotion_cnn.pt"
    mod_path = Path(__file__).parent / "models/emotion/classifier.py"
    
    try:
        emotion_mod = load_module(mod_path, "classifier")
        CLASS_LABELS = getattr(emotion_mod, "CLASS_LABELS", None)
        
        labels_path = project_root / "weights/labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                CLASS_LABELS = json.load(f)
        
        if CLASS_LABELS is None:
            CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # use gpu if available, otherwise cpu
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        model = emotion_mod.ConvNet(num_classes=len(CLASS_LABELS))
        ckpt = torch.load(str(model_path), map_location=device)
        
        state_dict = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
        if state_dict is None:
            state_dict = ckpt
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_ok = True
    except Exception as e:
        model_ok = False
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    try:
        eye_tracker = EyeTracker()
        eye_ok = True
        eye_err = None
    except Exception as e:
        eye_tracker = None
        eye_ok = False
        eye_err = str(e)


# tries to find a working camera at index 0, 1, or 2
def find_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, idx
            cap.release()
    return None, -1


# main function that analyzes each frame from the camera
def process_frame(frame):
    global stats, face_cascade, model, eye_tracker, device, CLASS_LABELS
    global Qtable,Ntable, eps, prev_st, prev_st_id, prev_act
    global problems, curr_issue,frame_cnt,last_intrpt, interrupt_msg
    
    if model is None or face_cascade is None:
        return frame
    
    frame_cnt += 1
    
    # get speech stats from the speech detection module
    wpm, filler, silence = 150.0, 0.05, 0.2
    try:
        wpm, filler, silence = get_speech_state()
        stats["wpm"] = int(wpm)
        stats["fillerWords"] = int(filler * 100)
        stats["silence"]= round(silence, 1)
    except Exception as e:
        pass
    
    # get eye contact score from eye tracker
    eye_contact_score = 1.0
    if eye_tracker is not None:
        try:
            result = eye_tracker.process_frame(frame.copy())
            if result and "prob_focused" in result:
                eye_contact_score = result["prob_focused"]
                stats["eyeContact"] = int(eye_contact_score * 100)
        except Exception as e:
            pass
    
    # detect faces and run emotion classification
    nervous_score = 0.0
    saw_face = False
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        saw_face = len(faces) > 0
        
        for (x, y, w, h) in faces:
            pad_w = int(0.15 * w)
            pad_h= int(0.2 * h)
            x1=max(0, x - pad_w)
            y1 =max(0, y-pad_h)
            x2 =min(frame.shape[1], x+w+pad_w)
            y2 = min(frame.shape[0], y +h+ pad_h)
            face = frame[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_array = face_resized.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_array).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(face_tensor)
                probs =F.softmax(logits, dim=1)
                idx= int(torch.argmax(probs, dim=1).item())
                conf= float(probs[0, idx].item())
            label =CLASS_LABELS[idx]
            nervous_score= compute_nervous_score(label, conf)
            stats["emotion"] = label
            stats["confidence"] =round(conf, 2)
            stats["nervousness"]=int(nervous_score * 100)
    except Exception as e:
        pass
    
    # skip rl decisions for first 100 frames while things stabilize
    warmup = 100
    if problems is not None and frame_cnt > warmup:
        try:
            if saw_face:
                update_consecutive_issue_counters(problems["nervous"], nervous_score >= nerv_thresh)
                update_consecutive_issue_counters(problems["eye_contact"], eye_contact_score < eye_thresh)
                update_consecutive_issue_counters(problems["speech_fast"], wpm >= fast_talk_thresh)
                update_consecutive_issue_counters(problems["speech_slow"], wpm > 0 and wpm <= slow_talk_thresh)
                update_consecutive_issue_counters(problems["filler"], filler >= filler_thresh)
                update_consecutive_issue_counters(problems["silence"], silence >= silence_thresh)
                update_consecutive_issue_counters(problems["no_face"], False)
            else:
                update_consecutive_issue_counters(problems["no_face"], True)
                for key in ["nervous", "eye_contact", "speech_fast", "speech_slow", "filler", "silence"]:
                    update_consecutive_issue_counters(problems[key], False)
            
            # rl agent decides whether to show an interruption
            if Qtable is not None:
                curr_st=np.array([nervous_score, eye_contact_score, wpm, filler, silence], dtype=float)
                curr_st_id=hash_state(curr_st)
                if prev_st is not None and prev_act is not None and prev_st_id is not None:
                    reward = compute_reward(prev_st, curr_st, prev_act)
                    update_Q(Qtable, Ntable, prev_st_id, curr_st_id, prev_act, reward, GAMMA)
                
                action = epsilon_greedy_action(Qtable, curr_st_id, num_actions=NUM_ACTIONS, epsilon=eps)
                eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
                applied_act = 0
                cooldown = 30
                can_intrpt = frame_cnt - last_intrpt >= cooldown
                if curr_issue is None and action == 1 and can_intrpt:
                    for key in issue_order:
                        if problems[key]["is_active"]:
                            curr_issue = key
                            applied_act = 1
                            last_intrpt= frame_cnt
                            interrupt_msg ={
                                "active": True,
                                "type": key,
                                "message": problems[key]["message"]
                            }
                            break
                else:
                    if curr_issue is not None:
                        applied_act = 1
                if curr_issue is not None:
                    if not problems[curr_issue]["is_active"]:
                        curr_issue = None
                        interrupt_msg = {"active": False, "type": None, "message": None}
                prev_st = curr_st
                prev_st_id = curr_st_id
                prev_act = applied_act
        except Exception as e:
            pass
    return frame

# generator that yields jpeg frames for the video stream
def gen_frames():
    global cam, running, last_frame, fr_lock
    while running:
        if cam is None:
            break
        ret, frame= cam.read()
        if not ret:
            break
        with fr_lock:
            last_frame=frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# runs in separate thread to process frames without blocking video
def processing_loop():
    global running, last_frame, fr_lock
    
    while running:
        frm = None
        with fr_lock:
            if last_frame is not None:
                frm= last_frame.copy()
        if frm is not None:
            process_frame(frm)
        time.sleep(0.05)


# starts the interview session opens camera and speech detection
@app.route('/start', methods=['POST'])
def start():
    global cam, running, speech_thrd, proc_thrd, last_frame, cam_idx
    global Qtable, Ntable, eps, prev_st, prev_st_id, prev_act
    global problems, curr_issue, frame_cnt, last_intrpt, interrupt_msg
    if running:
        return jsonify({"status": "already running"})
    
    cam, cam_idx = find_camera()
    if cam is None:
        return jsonify({"error": "No camera found. Please check camera permissions and make sure no other app is using it."}), 500
    
    running = True
    last_frame = None
    
    Qtable, Ntable = load_Q()
    eps = EPSILON_START
    prev_st= None
    prev_st_id =None
    prev_act= None
    problems = initialize_issues_dict()
    curr_issue = None
    frame_cnt = 0
    last_intrpt = 0
    interrupt_msg = {"active": False, "type": None, "message": None}
    
    proc_thrd = threading.Thread(target=processing_loop, daemon=True)
    proc_thrd.start()
    
    stt_stop_event.clear()
    speech_thrd= start_speech_in_background()
    
    return jsonify({
        "status": "started",
        "camera_index": cam_idx,
        "eye_tracker": eye_ok,
        "model_loaded": model_ok
    })

@app.route('/stop', methods=['POST'])
def stop():
    global cam, running, speech_thrd, Qtable, Ntable
    
    running=False
    
    if cam:
        cam.release()
        cam= None
    try:
        if Qtable is not None:
            save_Q(Qtable, Ntable)
    except:
        pass
    try:
        stt_stop_event.set()
        if speech_thrd:
            speech_thrd.join(timeout=2)
    except:
        pass
    return jsonify({"status": "stopped"})


# streams video to the frontend as mjpeg
@app.route('/video')
def video():
    if not running:
        return "not running", 400
    response = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/interruption')
def interruption():
    return jsonify(interrupt_msg)

@app.route('/transcript')
def transcript():
    try:
        path=Path(__file__).parent / "models/speech/transcript.txt"
        if path.exists():
            with open(path) as f:
                content = f.read()
            
            sentences =[]
            for line in content.split('\n'):
                line =line.strip()
                if line.startswith('AssemblyAI:'):
                    text=line.replace('AssemblyAI:', '').strip()
                    if text and len(text) > 2:
                        sentences.append(text)
            
            return jsonify({"lines": sentences[-15:]})
    except:
        pass
    return jsonify({"lines": []})


@app.route('/health')
def health():
    test_cam, test_idx = find_camera()
    cam_works=test_cam is not None
    if test_cam:
        test_cam.release()
    result = {
        "status":"ok",
        "camera_available": cam_works,
        "camera_index": test_idx if cam_works else None,
        "eye_tracker_available": eye_ok,
        "model_loaded": model_ok,
        "is_running": running
    }
    if not eye_ok and eye_err:
        result["eye_tracker_error"] = eye_err
    return jsonify(result)

# start the server on port 6767
if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=6767, debug=False, threaded=True)