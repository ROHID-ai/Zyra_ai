"""
Common detection representation and unified live scene state.

All vision modules (YOLO, currency, future stairs/door/OCR) should normalize
into DetectionRecord before entering spatial / safety / event pipelines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectionRecord:
    """Normalized detection from any vision source."""

    name: str
    confidence: float = 0.0
    bbox: Optional[List[int]] = None
    center: Optional[List[float]] = None
    area: float = 0.0
    tracked_id: Optional[int] = None
    stable: bool = False
    source: str = "object"  # object | currency | stairs | door | ocr
    position: str = "center"
    distance: str = "medium"
    motion: str = "stationary"
    priority: int = 3
    kind: str = "object"
    reason: str = "detected"
    in_path: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["extra"]:
            d.pop("extra", None)
        return d


def from_yolo_detection(det: dict) -> DetectionRecord:
    """Convert raw YOLO/spatial detection dict to common record."""
    name = str(det.get("name") or det.get("class") or "object")
    return DetectionRecord(
        name=name,
        confidence=float(det.get("confidence") or 0.0),
        bbox=det.get("bbox"),
        center=det.get("center"),
        area=float(det.get("area") or 0.0),
        tracked_id=det.get("tracked_id"),
        stable=bool(det.get("stable")),
        source="object",
        position=str(det.get("position") or "center"),
        distance=str(det.get("distance") or "medium"),
        motion=str(det.get("motion") or "stationary"),
        priority=int(det.get("priority", 3)),
        kind=str(det.get("kind") or "object"),
        reason=str(det.get("reason") or "detected"),
        in_path=bool(det.get("in_path")),
    )


def apply_safety_fields(record: DetectionRecord, ranked: dict) -> DetectionRecord:
    """Merge SafetyEngine output into a DetectionRecord."""
    record.name = str(ranked.get("name") or record.name)
    record.priority = int(ranked.get("priority", record.priority))
    record.kind = str(ranked.get("kind") or record.kind)
    record.reason = str(ranked.get("reason") or record.reason)
    record.in_path = bool(ranked.get("in_path"))
    record.position = str(ranked.get("position") or record.position)
    record.distance = str(ranked.get("distance") or record.distance)
    record.motion = str(ranked.get("motion") or record.motion)
    record.confidence = float(ranked.get("confidence") or record.confidence)
    record.tracked_id = ranked.get("tracked_id", record.tracked_id)
    record.stable = bool(ranked.get("stable", record.stable))
    return record


@dataclass
class LiveState:
    """Single structured snapshot of the current environment."""

    mode: Optional[str] = None
    objects: List[Dict[str, Any]] = field(default_factory=list)
    hazards: List[Dict[str, Any]] = field(default_factory=list)
    path: Dict[str, Any] = field(default_factory=lambda: {"status": "clear"})
    danger: str = "low"
    hazard: Optional[Dict[str, Any]] = None
    currency: Dict[str, Any] = field(
        default_factory=lambda: {
            "currency": [],
            "total": 0,
            "signature": "",
            "spoken": "",
        }
    )
    environment: str = "unknown"
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    last_response: str = ""
    recent_detections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "objects": list(self.objects),
            "hazards": list(self.hazards),
            "path": dict(self.path),
            "danger": self.danger,
            "hazard": dict(self.hazard) if self.hazard else None,
            "currency": dict(self.currency),
            "environment": self.environment,
            "recent_events": list(self.recent_events),
            "last_response": self.last_response,
            "recent_detections": list(self.recent_detections),
        }

    def update_object_scene(
        self,
        safety: Dict[str, Any],
        mode: Optional[str],
        recent_detections: Optional[List[Dict[str, Any]]] = None,
        recent_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Apply SafetyEngine output to live state."""
        self.mode = mode
        ranked = safety.get("objects") or []
        self.objects = [
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
            for o in ranked
        ]
        self.hazards = [
            {
                "name": h.get("name"),
                "position": h.get("position"),
                "distance": h.get("distance"),
                "motion": h.get("motion"),
                "priority": h.get("priority"),
                "kind": h.get("kind"),
                "reason": h.get("reason"),
            }
            for h in (safety.get("hazards") or [])
        ]
        self.path = dict(safety.get("path") or {"status": "clear"})
        self.danger = str(safety.get("danger") or "low")
        hazard = safety.get("hazard")
        self.hazard = (
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
        self.environment = _infer_environment(ranked)
        if recent_detections is not None:
            self.recent_detections = list(recent_detections)
        if recent_events is not None:
            self.recent_events = list(recent_events)

    def update_currency_scene(
        self,
        summary: Dict[str, Any],
        mode: Optional[str],
        recent_detections: Optional[List[Dict[str, Any]]] = None,
        recent_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.mode = mode
        self.currency = dict(summary)
        self.objects = []
        self.hazards = []
        self.path = {"status": "clear"}
        self.danger = "low"
        self.hazard = None
        self.environment = "indoor"
        if recent_detections is not None:
            self.recent_detections = list(recent_detections)
        if recent_events is not None:
            self.recent_events = list(recent_events)

    def reset_scene(self, keep_mode: bool = True) -> None:
        if not keep_mode:
            self.mode = None
        self.objects = []
        self.hazards = []
        self.path = {"status": "clear"}
        self.danger = "low"
        self.hazard = None
        self.recent_detections = []
        if not keep_mode:
            self.currency = {
                "currency": [],
                "total": 0,
                "signature": "",
                "spoken": "",
            }


def _infer_environment(objects: List[dict]) -> str:
    """Light heuristic — no dedicated classifier yet."""
    outdoor = {
        "car",
        "motorcycle",
        "bicycle",
        "bus",
        "truck",
        "traffic light",
        "stop sign",
    }
    names = {str(o.get("name") or "").lower() for o in objects}
    if names & outdoor:
        return "outdoor"
    if objects:
        return "indoor"
    return "unknown"


def detections_for_display(detections: List[dict]) -> List[Dict[str, Any]]:
    """Build UI-friendly recent_detections from enriched YOLO output."""
    return [
        {
            "label": d.get("name") or d.get("class"),
            "confidence": round(float(d.get("confidence") or 0) * 100),
            "position": d.get("position"),
            "distance": d.get("distance"),
            "motion": d.get("motion"),
        }
        for d in detections
    ]
