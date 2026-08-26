"""FastAPI server for Zyra AI — real-time vision assistant for blind users."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

# Ensure relative paths (weights/, logs/, system_config.json) resolve correctly
os.chdir(Path(__file__).resolve().parent)

import cv2
import speech_recognition as sr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from camera import ThreadedCamera
from config import ConfigManager
from currency_detection import CurrencyDetector
from event_engine import EventEngine
from groq_service import GroqService
from object_detection import ObjectDetector
from preprocessing import FramePreprocessor
from safety_engine import SafetyEngine
from spatial_analyzer import SpatialAnalyzer
from voice_engine import VoiceEngine

DetectionMode = Optional[Literal["object", "currency"]]


class ConfidenceUpdate(BaseModel):
    mode: Optional[str] = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CameraScaleUpdate(BaseModel):
    scale: float = Field(default=0.75, ge=0.3, le=1.0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def _number_words(n: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(n, str(n))


def _rupee_words(value: int) -> str:
    mapping = {
        10: "ten",
        20: "twenty",
        50: "fifty",
        100: "one-hundred",
        200: "two-hundred",
        500: "five-hundred",
    }
    return mapping.get(value, str(value))


def build_currency_summary(detections: list[dict]) -> dict:
    """Aggregate visible notes into counts + spoken total (live frame state)."""
    counts: Counter[int] = Counter()
    for det in detections:
        denom = int(det.get("denomination") or 0)
        if denom:
            counts[denom] += 1

    currency = [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), reverse=True)
    ]
    total = sum(v * c for v, c in counts.items())
    if not currency:
        return {
            "currency": [],
            "total": 0,
            "signature": "",
            "spoken": "",
        }

    parts = []
    for item in currency:
        value = item["value"]
        count = item["count"]
        note = f"{_rupee_words(value)}-rupee note"
        if count > 1:
            parts.append(f"{_number_words(count)} {note}s")
        else:
            parts.append(f"one {note}")

    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        joined = f"{parts[0]} and {parts[1]}"
    else:
        joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    spoken = f"I can see {joined}. Total {total} rupees."
    signature = "|".join(f"{v}x{c}" for v, c in sorted(counts.items()))
    return {
        "currency": currency,
        "total": total,
        "signature": signature,
        "spoken": spoken,
    }


class VisionService:
    """Shared vision pipeline state — live camera → detect → understand → speak."""

    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.camera: Optional[ThreadedCamera] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.currency_detector: Optional[CurrencyDetector] = None
        self.voice_engine: Optional[VoiceEngine] = None
        self.preprocessor: Optional[FramePreprocessor] = None
        self.spatial = SpatialAnalyzer()
        self.safety = SafetyEngine()
        self.events = EventEngine()
        self.groq = GroqService()
        self.current_mode: DetectionMode = None
        self._voice_thread: Optional[threading.Thread] = None
        self._voice_running = False
        self._ready = False
        self._init_error: Optional[str] = None
        self._state_lock = threading.Lock()
        self.last_detections: list[dict] = []
        self.live_objects: list[dict] = []
        self.live_path: dict = {"status": "clear"}
        self.live_danger: str = "low"
        self.live_hazard: Optional[dict] = None
        self.currency_summary: dict = {
            "currency": [],
            "total": 0,
            "signature": "",
            "spoken": "",
        }
        self.last_response: str = ""
        self.core_state: str = "idle"  # idle|listening|vision|thinking|warning|responding|offline
        self._frame_index = 0

    @property
    def is_ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        try:
            config = self.config_manager.get_config()

            print("\n[Vision System] Initializing components...")

            spatial_cfg = config.spatial
            self.spatial.configure(
                left_max=spatial_cfg.left_max,
                right_min=spatial_cfg.right_min,
                near_area_ratio=spatial_cfg.near_area_ratio,
                far_area_ratio=spatial_cfg.far_area_ratio,
            )
            self.safety.path_band_top = spatial_cfg.path_band_top

            self.camera = ThreadedCamera(
                camera_id=config.camera.camera_id,
                fps=config.camera.fps,
                resize_scale=config.camera.resize_scale,
            )
            self.camera.start()

            self.object_detector = ObjectDetector(
                model_path=config.object_detection.model_path,
                conf_threshold=config.object_detection.conf_threshold,
                nms_threshold=config.object_detection.nms_threshold,
            )

            self.currency_detector = CurrencyDetector(
                model_path=config.currency_detection.model_path,
                use_custom_model=config.currency_detection.use_custom_model,
            )

            self.voice_engine = VoiceEngine(
                rate=config.voice.rate,
                volume=config.voice.volume,
            )
            self.voice_engine.set_cooldowns(
                global_cooldown=config.voice.global_cooldown,
                object_cooldown=config.voice.object_cooldown_duration,
            )

            self.preprocessor = FramePreprocessor(
                enable_adaptive_histogram=config.preprocessing.enhance_contrast,
            )

            self._voice_running = True
            self._voice_thread = threading.Thread(
                target=self._voice_listener, daemon=True
            )
            self._voice_thread.start()

            self._ready = True
            self.core_state = "idle"
            print("[Vision System] All components initialized successfully")
        except Exception as exc:
            self._init_error = str(exc)
            self.core_state = "offline"
            print(f"[Vision System] Initialization failed: {exc}")

    def shutdown(self) -> None:
        self._voice_running = False
        if self.camera:
            self.camera.stop()
        if self.voice_engine:
            self.voice_engine.shutdown()
        print("[Vision System] Shutdown complete")

    def change_mode(self, mode: DetectionMode) -> None:
        if mode == self.current_mode:
            return

        self.current_mode = mode
        self.config_manager.set_mode(mode)
        self.events.reset()
        self.spatial.clear()

        with self._state_lock:
            self.live_objects = []
            self.live_path = {"status": "clear"}
            self.live_danger = "low"
            self.live_hazard = None
            self.last_detections = []
            if mode != "currency":
                self.currency_summary = {
                    "currency": [],
                    "total": 0,
                    "signature": "",
                    "spoken": "",
                }

        if self.voice_engine:
            if mode == "object":
                msg = "Vision activated. I will warn you about important changes."
                self.core_state = "vision"
            elif mode == "currency":
                msg = "Currency mode activated."
                self.core_state = "vision"
            else:
                msg = "Vision stopped."
                self.core_state = "idle"
            self.last_response = msg
            self.voice_engine.announce_priority(msg)

    def restart_camera(self, scale: float) -> None:
        config = self.config_manager.get_config()
        if self.camera:
            self.camera.stop()
        self.camera = ThreadedCamera(
            camera_id=config.camera.camera_id,
            fps=config.camera.fps,
            resize_scale=scale,
        )
        self.camera.start()

    def get_live_state(self) -> dict:
        with self._state_lock:
            return {
                "mode": self.current_mode,
                "objects": list(self.live_objects),
                "path": dict(self.live_path) if isinstance(self.live_path, dict) else self.live_path,
                "danger": self.live_danger,
                "hazard": self.live_hazard,
                "currency": dict(self.currency_summary),
                "last_response": self.last_response,
            }

    def ask(self, question: str) -> str:
        """Answer from CURRENT live detection state — no frame capture."""
        self.core_state = "thinking"
        live = self.get_live_state()
        answer = self.groq.answer_question(question, live)
        self.last_response = answer
        self.core_state = "responding"
        if self.voice_engine:
            self.voice_engine.announce_priority(answer, object_key="ask_zyra")
        return answer

    def handle_utterance(self, utterance: str) -> str:
        intent = self.groq.classify_intent(utterance)
        action = intent.get("action", "unknown")
        query = intent.get("query") or utterance

        if action == "start_object":
            self.change_mode("object")
            return self.last_response or "Vision activated."
        if action == "start_currency":
            self.change_mode("currency")
            return self.last_response or "Currency mode activated."
        if action == "stop":
            self.change_mode(None)
            return self.last_response or "Vision stopped."
        if action == "help":
            msg = (
                "You can say start vision, check currency, stop, "
                "what do you see, what's ahead, or where is my phone."
            )
            self.last_response = msg
            if self.voice_engine:
                self.voice_engine.announce_priority(msg)
            return msg
        if action == "ask":
            return self.ask(query)

        # Fallback: treat as a question if it looks like one
        lower = utterance.lower()
        if "?" in utterance or any(
            w in lower for w in ("what", "where", "is there", "how many", "anyone")
        ):
            return self.ask(utterance)

        msg = "I did not catch that. Try asking what I see, or say start vision."
        self.last_response = msg
        if self.voice_engine:
            self.voice_engine.announce_priority(msg)
        return msg

    def _voice_listener(self) -> None:
        recognizer = sr.Recognizer()
        print("[Voice Recognition] Initializing...")
        try:
            mic = sr.Microphone()
        except Exception as exc:
            print(f"[Voice Recognition] No microphone ({exc}). Voice commands disabled.")
            return

        while self._voice_running:
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.35)
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=4)
                cmd = recognizer.recognize_google(audio).lower()
                print(f"[Voice Cmd] {cmd}")
                self.core_state = "listening"
                self.handle_utterance(cmd)
            except Exception:
                pass
            time.sleep(0.35)

    def _speak_events(self, events: list[dict], live_state: dict) -> None:
        if not self.voice_engine or not events:
            return

        # Speak only the highest-priority event this tick to avoid spam
        top = events[0]
        message = top.get("message") or ""
        critical = bool(top.get("critical"))

        if not critical and self.groq.enabled and top.get("priority", 9) >= 2:
            # Non-blocking polish is skipped for latency; keep local wording.
            pass

        spoken = self.voice_engine.announce_event(
            message,
            critical=critical,
            object_key=top.get("key") or top.get("type"),
        )
        if spoken:
            self.last_response = message
            if critical:
                self.core_state = "warning"
            else:
                self.core_state = "responding"

    def _process_object_frame(self, frame, config) -> tuple:
        detections = self.object_detector.detect(
            frame,
            conf_threshold=config.object_detection.conf_threshold,
        )
        h, w = frame.shape[:2]
        enriched = self.spatial.enrich(
            detections,
            frame_width=w,
            frame_height=h,
            track_history=self.object_detector.track_history,
        )
        # Prefer stable tracks for announcements; keep all for UI overlay
        stable = [d for d in enriched if d.get("stable")]
        safety = self.safety.analyze(stable or enriched, w, h)
        events = self.events.process_scene(safety)

        with self._state_lock:
            self.live_objects = [
                {
                    "name": o.get("name"),
                    "class": o.get("class") or o.get("name"),
                    "position": o.get("position"),
                    "distance": o.get("distance"),
                    "motion": o.get("motion"),
                    "confidence": round(float(o.get("confidence") or 0) * 100),
                    "tracked_id": o.get("tracked_id"),
                    "priority": o.get("priority"),
                    "kind": o.get("kind"),
                    "in_path": o.get("in_path"),
                    "stable": o.get("stable"),
                }
                for o in safety.get("objects") or []
            ]
            self.live_path = safety.get("path") or {"status": "clear"}
            self.live_danger = safety.get("danger") or "low"
            hazard = safety.get("hazard")
            self.live_hazard = (
                {
                    "name": hazard.get("name"),
                    "reason": hazard.get("reason"),
                    "priority": hazard.get("priority"),
                    "position": hazard.get("position"),
                    "distance": hazard.get("distance"),
                    "motion": hazard.get("motion"),
                }
                if hazard
                else None
            )
            self.last_detections = [
                {
                    "label": d.get("name") or d.get("class"),
                    "confidence": round(float(d.get("confidence") or 0) * 100),
                    "position": d.get("position"),
                    "distance": d.get("distance"),
                    "motion": d.get("motion"),
                }
                for d in (stable or enriched)
            ]

        live = self.get_live_state()
        self._speak_events(events, live)

        if self.live_danger in ("critical", "high"):
            self.core_state = "warning"
        elif self.current_mode and not self.live_objects:
            self.core_state = "thinking"
        elif self.current_mode:
            self.core_state = "vision"

        annotated = self.object_detector.draw_detections(
            frame, enriched, show_confidence=True
        )
        return annotated, enriched

    def _process_currency_frame(self, frame, config) -> tuple:
        detections = self.currency_detector.detect(
            frame,
            conf_threshold=config.currency_detection.conf_threshold,
        )
        summary = build_currency_summary(detections)
        safety = {
            "objects": [],
            "path": {"status": "clear"},
            "danger": "low",
            "hazard": None,
        }
        events = self.events.process_scene(safety, currency_summary=summary)

        with self._state_lock:
            self.currency_summary = summary
            self.live_objects = []
            self.live_path = {"status": "clear"}
            self.live_danger = "low"
            self.live_hazard = None
            self.last_detections = [
                {
                    "label": f"₹{d['denomination']}",
                    "confidence": round(float(d.get("confidence") or 0) * 100),
                }
                for d in detections
            ]

        self._speak_events(events, self.get_live_state())
        if self.current_mode:
            self.core_state = "vision" if detections else "thinking"

        annotated = self.currency_detector.draw_detections(
            frame, detections, show_confidence=True
        )
        return annotated, detections

    def frame_generator(self) -> Iterator[bytes]:
        while True:
            if not self.camera or not self.preprocessor:
                time.sleep(0.1)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                continue

            config = self.config_manager.get_config()
            frame = self.preprocessor.preprocess_for_detection(
                frame,
                self.config_manager.get_preprocessing_config(),
            )

            self._frame_index += 1

            if self.current_mode == "object" and self.object_detector:
                frame, _ = self._process_object_frame(frame, config)
            elif self.current_mode == "currency" and self.currency_detector:
                frame, _ = self._process_currency_frame(frame, config)
            else:
                with self._state_lock:
                    self.last_detections = []
                    self.live_objects = []

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    def get_status(self) -> dict:
        logs: list[str] = []
        log_file = self.config_manager.get_config().log_file
        try:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as handle:
                    logs = handle.readlines()[-8:]
        except OSError:
            pass

        live = self.get_live_state()
        path = live.get("path") or {}
        path_status = path.get("status", "clear") if isinstance(path, dict) else path

        return {
            # Existing fields (keep compatible)
            "mode": self.current_mode,
            "logs": logs,
            "camera_stats": self.camera.get_stats() if self.camera else {},
            "object_detector_stats": (
                self.object_detector.get_stats() if self.object_detector else {}
            ),
            "currency_detector_stats": (
                self.currency_detector.get_stats() if self.currency_detector else {}
            ),
            "voice_engine_stats": (
                self.voice_engine.get_stats() if self.voice_engine else {}
            ),
            "recent_detections": self.last_detections,
            # Extended accessibility / live-scene fields
            "objects": live.get("objects") or [],
            "path": path_status,
            "path_detail": path,
            "hazard": live.get("hazard"),
            "danger": live.get("danger") or "low",
            "currency": live.get("currency") or {},
            "recent_events": list(self.events.recent_events[-10:]),
            "last_response": self.last_response
            or (self.voice_engine.last_announcement if self.voice_engine else ""),
            "core_state": self.core_state,
            "groq": self.groq.get_status(),
            "camera": self.camera.get_stats() if self.camera else {},
            "voice": self.voice_engine.get_stats() if self.voice_engine else {},
        }


vision = VisionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_thread = threading.Thread(target=vision.initialize, daemon=True)
    init_thread.start()
    yield
    vision.shutdown()


app = FastAPI(title="Zyra AI Vision API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "ready": vision.is_ready,
        "error": vision._init_error or "",
    }


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        vision.frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/start-object")
def start_object() -> JSONResponse:
    vision.change_mode("object")
    return JSONResponse({"status": "Object detection started"})


@app.get("/start-currency")
def start_currency() -> JSONResponse:
    vision.change_mode("currency")
    return JSONResponse({"status": "Currency detection started"})


@app.get("/stop")
def stop_detection() -> JSONResponse:
    vision.change_mode(None)
    return JSONResponse({"status": "Detection stopped"})


@app.get("/status")
def get_status() -> JSONResponse:
    return JSONResponse(vision.get_status())


@app.get("/config")
def get_config() -> JSONResponse:
    return JSONResponse(json.loads(vision.config_manager.get_json()))


@app.post("/config/set-confidence")
def set_confidence(body: ConfidenceUpdate) -> JSONResponse:
    mode = body.mode or vision.current_mode
    vision.config_manager.set_confidence_threshold(mode, body.threshold)
    return JSONResponse({"status": "Confidence threshold updated"})


@app.post("/config/set-camera-scale")
def set_camera_scale(body: CameraScaleUpdate) -> JSONResponse:
    vision.config_manager.set_camera_scale(body.scale)
    vision.restart_camera(body.scale)
    return JSONResponse({"status": "Camera scale updated"})


@app.post("/ask")
def ask_zyra(body: AskRequest) -> JSONResponse:
    answer = vision.ask(body.question)
    return JSONResponse({"answer": answer, "live_state": vision.get_live_state()})


@app.post("/command")
def voice_command(body: AskRequest) -> JSONResponse:
    """Natural-language command / question using live state."""
    result = vision.handle_utterance(body.question)
    return JSONResponse({"result": result, "mode": vision.current_mode})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
