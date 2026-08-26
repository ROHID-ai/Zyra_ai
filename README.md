# Vision AI — Zyra

Real-time AI vision assistant for blind and visually impaired users.

**SEE → UNDERSTAND → PRIORITIZE → SPEAK**

Live camera is processed continuously with YOLO/OpenCV. Groq is an optional language layer over structured live state — never a frame-by-frame detector.

## Project structure

```
II/
├── backend/                 # FastAPI + vision pipeline
│   ├── main.py              # API + VisionService
│   ├── camera.py
│   ├── object_detection.py
│   ├── currency_detection.py
│   ├── spatial_analyzer.py  # LEFT/CENTER/RIGHT, NEAR/MEDIUM/FAR, motion
│   ├── safety_engine.py     # hazards + path awareness
│   ├── event_engine.py      # meaningful state-change events
│   ├── groq_service.py      # optional NL / Q&A (backend-only)
│   ├── preprocessing.py
│   ├── voice_engine.py
│   ├── config.py
│   ├── system_config.json
│   ├── requirements.txt
│   ├── .env.example         # GROQ_API_KEY=
│   ├── weights/             # yolo11n.pt, best.pt
│   └── logs/
├── frontend/                # Next.js Zyra UI
│   ├── src/app/
│   ├── src/components/
│   └── src/lib/
├── notebooks/
│   └── Vision_AI_Colab.ipynb
└── README.md
```

## Quick start

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: set GROQ_API_KEY
python main.py
```

API: **http://localhost:8000**

### Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

UI: **http://localhost:3000**

## Real-time pipeline

```
LIVE CAMERA → preprocess → YOLO / currency → tracking
  → position / distance / motion → safety → events
  → voice priority → TTS

User questions → structured LIVE state → (optional Groq) → answer
```

Critical safety warnings speak **immediately** via the local event/voice engine. They never wait on Groq.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/video_feed` | MJPEG stream with detections |
| GET | `/start-object` | Start object / vision mode |
| GET | `/start-currency` | Start currency detection |
| GET | `/stop` | Stop detection |
| GET | `/status` | System + live scene (objects, path, hazard, events) |
| POST | `/ask` | Ask Zyra using current live state |
| POST | `/command` | Natural language command / question |
| GET | `/config` | Current configuration |
| POST | `/config/set-confidence` | Update confidence threshold |
| POST | `/config/set-camera-scale` | Update camera scale |

## Controls

| Key / Voice | Action |
|-------------|--------|
| `1` / "start vision" | Object / vision mode |
| `2` / "currency" | Currency detection |
| `0` / "stop" | Stop |
| "What do you see?" | Answer from live state |
| "What's ahead?" / "Where is my phone?" | Spatial Q&A |

## Accessibility principles

- Audio is primary; UI mirrors spoken information
- Conservative wording ("appears clear", never "safe to walk")
- Event-based speech (not every frame)
- Vehicles / path obstacles / approaching people prioritized

## Requirements

- Python 3.10+
- Node.js 20+
- Webcam
- GPU optional (CUDA speeds up YOLO)
- Optional `GROQ_API_KEY` for richer Q&A wording

## Colab

See `notebooks/Vision_AI_Colab.ipynb` for a standalone Google Colab runner.
