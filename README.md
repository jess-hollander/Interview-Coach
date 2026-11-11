# interview coach

## setup

### prerequisites

- python 3.11 (mediapipe doesnt work on 3.12+)
- node.js
- conda (optional but recomended)

### 1. clone and setup environment

```bash
git clone https://github.com/annahiggins1/interview-coach.git
cd interview-coach
conda create -n interview_coach python=3.11 -y
conda activate interview_coach
```

### 2. install dependencies

```bash
python -m pip install -r requirements.txt
cd frontend && npm install && cd ..
```

## running it

need 2 terminals:

**terminal 1 - backend:**
```bash
conda activate interview_coach
# Set ASSEMBLYAI_API_KEY in the project-root .env file (see .env.example)
cd backend
python api.py
```

**terminal 2 - frontend:**
```bash
cd frontend
npm start
```

then go to http://localhost:3000

Create a project-root `.env` file containing `ASSEMBLYAI_API_KEY=your_key` before starting the backend.

## project structure

```
backend/
  api.py              # main flask server
  models/
    emotion/          # cnn for emotion classification
    eye_tracking/     # mediapipe eye tracker with hmm
    speech/           # speech to text and analysis
    rl/               # q-learning agent
frontend/
  src/
    App.js            # main react component
    components/       # webcam, stats, transcript, etc
weights/
  emotion_cnn.pt      # trained emotion model
data/
  train/              #  training images
  test/               #  test images
```

## troubleshooting

**camera not working:** make sure no other app is using your camera

**mediapipe error:** make sure your on python 3.11, not 3.12

**model not loading:** check that weights/emotion_cnn.pt exists

