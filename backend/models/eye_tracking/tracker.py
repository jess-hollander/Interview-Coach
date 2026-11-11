# eye tracking stuff using hmm and mediapipe
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# hidden markov model for smoothing out the eye tracking predictions
class HMM:
    # sets up the hmm with states, observations and probabilities
    def __init__(self,states,observations,start_prob,trans_prob,emit_prob):
        self.states=states
        self.observations=observations
        self.start_prob=start_prob
        self.trans_prob=trans_prob
        self.emit_prob=emit_prob
        self.current_state_probs=np.array(start_prob)

    # updates probabilities based on what we just saw
    def forward_step(self,observation):
        obs_idx=self.observations.index(observation)
        new_probs=np.zeros(len(self.states))

        for j,state_to in enumerate(self.states):
            prob_sum=0
            for i,state_from in enumerate(self.states):
                prob_sum +=self.current_state_probs[i]*self.trans_prob[i][j]
            new_probs[j]=prob_sum*self.emit_prob[j][obs_idx]

        # normalize so probabilities add up to 1
        total=np.sum(new_probs)
        if total >0:
            new_probs/=total

        self.current_state_probs =new_probs
        return self.states[np.argmax(new_probs)],new_probs

    # resets the hmm back to initial state
    def reset(self):
        self.current_state_probs=np.array(self.start_prob)


# main eye tracker class, uses mediapipe face mesh
class EyeTracker:
    def __init__(self):
        # stores last 15 frames of focus probability
        self.history=deque(maxlen=15)
        self.mp_face=mp.solutions.face_mesh
        self.face_mesh=self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # mediapipe landmark indices for eyes and face
        self.LEFT_IRIS=[468,469,470,471,472]
        self.RIGHT_IRIS=[473,474,475,476,477]
        self.LEFT_EYE_H=[33,133]
        self.RIGHT_EYE_H=[362,263]
        self.NOSE_TIP=1
        self.CHIN=152
        self.FOREHEAD=10

        # baseline head pitch gets set during calibration
        self.baseline_pitch=None
        self.calibration_frames=0

        # hmm states and transition probabilities
        states=["FOCUSED","DISTRACTED"]
        observations=["HIGH","MEDIUM","LOW"]

        start_prob=[0.7,0.3]

        trans_prob=[
            [0.8,0.2],
            [0.4,0.6]
        ]

        emit_prob=[
            [0.7,0.25,0.05],
            [0.05,0.25,0.7]
        ]

        self.hmm=HMM(states,observations,start_prob,trans_prob,emit_prob)

    # figures out where the iris is horizontally in the eye
    def get_iris_horizontal(self,lm,iris,eye_h):
        ix=np.mean([lm[i].x for i in iris])
        lc,rc=lm[eye_h[0]],lm[eye_h[1]]
        w=abs(rc.x-lc.x)
        return (ix-lc.x)/w if w>0.01 else 0.5

    # checks if head is tilted up or down
    def get_head_pitch(self, lm):
        nose_tip= lm[self.NOSE_TIP]
        forehead = lm[self.FOREHEAD]
        chin = lm[self.CHIN]

        upper=abs(nose_tip.y-forehead.y)
        lower = abs(chin.y - nose_tip.y)

        if lower>0.01:
            pitch_ratio=upper/lower
        else:
            pitch_ratio = 1.0

        return pitch_ratio

    # turns the score into high/medium/low for the hmm
    def discretize_score(self,score):
        if score>=0.6:
            return "HIGH"
        elif score>=0.4:
            return "MEDIUM"
        else:
            return "LOW"

    # main function that processes each video frame
    def process_frame(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=self.face_mesh.process(rgb)

        # no face detected so assume distracted
        if not res.multi_face_landmarks:
            obs="LOW"
            state,probs=self.hmm.forward_step(obs)
            prob_focused=probs[0]
            self.history.append(prob_focused)
            avg=np.mean(list(self.history))
            return {"state":state,"prob_focused":avg,"score":avg*100}

        lm=res.multi_face_landmarks[0].landmark

        # get horizontal gaze from both eyes and average them
        lx=self.get_iris_horizontal(lm,self.LEFT_IRIS,self.LEFT_EYE_H)
        rx = self.get_iris_horizontal(lm, self.RIGHT_IRIS, self.RIGHT_EYE_H)
        gaze_x=(lx+rx)/2

        pitch=self.get_head_pitch(lm)

        # calibrate for first 15 frames to get baseline head position
        if self.calibration_frames <15:
            if self.baseline_pitch is None:
                self.baseline_pitch=pitch
            else:
                self.baseline_pitch=self.baseline_pitch*0.8+pitch*0.2
            self.calibration_frames +=1

        # how far eyes are from center horizontally
        x_dev=abs(gaze_x-0.5)
        if x_dev<0.08:
            x_dev = 0
        x_score =1.0-x_dev*4.0

        # how far head is tilted from baseline
        if self.baseline_pitch is not None:
            pitch_dev = abs(pitch - self.baseline_pitch)
            if pitch_dev<0.08:
                pitch_dev =0
            y_score = 1.0 - pitch_dev * 3.0
        else:
            y_score=1.0

        # combine x and y scores into one gaze score
        gaze_score=(x_score*0.6+y_score*0.4)
        gaze_score=max(0.1,min(1.0,gaze_score))

        # also check if head is centered
        nose_x=abs(lm[1].x-0.5)
        head_score = 1.0 - nose_x * 3
        head_score= max(0.1,min(1.0,head_score))

        raw_score=gaze_score*0.7+head_score*0.3

        obs = self.discretize_score(raw_score)
        state,probs = self.hmm.forward_step(obs)
        prob_focused=probs[0]

        # average over history to smooth out the result
        self.history.append(prob_focused)
        avg=np.mean(list(self.history))
        final_state="FOCUSED" if avg>0.5 else "DISTRACTED"

        return {"state":final_state,"prob_focused":avg,"score":avg*100}
