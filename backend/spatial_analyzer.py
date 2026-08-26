"""
Spatial enrichment for live detections: position, relative distance, motion.
Works on YOLO detections + existing track history — no separate capture loop.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple


Position = str  # left | center | right
Distance = str  # near | medium | far
Motion = str  # stationary | moving | approaching | moving_away | crossing_path


class SpatialAnalyzer:
    """Derive LEFT/CENTER/RIGHT, NEAR/MEDIUM/FAR, and motion from boxes."""

    # Typical relative area share of frame for common COCO classes (heuristic).
    # Used only to temper distance; never claimed as meters.
    CLASS_SIZE_HINT: Dict[str, float] = {
        "person": 0.18,
        "chair": 0.12,
        "couch": 0.28,
        "dining table": 0.25,
        "bottle": 0.03,
        "cup": 0.02,
        "backpack": 0.06,
        "handbag": 0.04,
        "suitcase": 0.08,
        "laptop": 0.06,
        "cell phone": 0.015,
        "book": 0.03,
        "bicycle": 0.2,
        "motorcycle": 0.22,
        "car": 0.35,
        "bus": 0.55,
        "truck": 0.5,
        "door": 0.2,
        "stairs": 0.25,
    }

    def __init__(
        self,
        left_max: float = 0.33,
        right_min: float = 0.66,
        near_area_ratio: float = 0.12,
        far_area_ratio: float = 0.03,
        history_len: int = 12,
        motion_pixel_threshold: float = 18.0,
        approach_area_growth: float = 1.18,
        retreat_area_shrink: float = 0.85,
        cross_min_dx_ratio: float = 0.12,
    ) -> None:
        self.left_max = left_max
        self.right_min = right_min
        self.near_area_ratio = near_area_ratio
        self.far_area_ratio = far_area_ratio
        self.motion_pixel_threshold = motion_pixel_threshold
        self.approach_area_growth = approach_area_growth
        self.retreat_area_shrink = retreat_area_shrink
        self.cross_min_dx_ratio = cross_min_dx_ratio
        self._histories: Dict[Any, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )

    def configure(
        self,
        left_max: Optional[float] = None,
        right_min: Optional[float] = None,
        near_area_ratio: Optional[float] = None,
        far_area_ratio: Optional[float] = None,
    ) -> None:
        if left_max is not None:
            self.left_max = left_max
        if right_min is not None:
            self.right_min = right_min
        if near_area_ratio is not None:
            self.near_area_ratio = near_area_ratio
        if far_area_ratio is not None:
            self.far_area_ratio = far_area_ratio

    def enrich(
        self,
        detections: List[dict],
        frame_width: int,
        frame_height: int,
        track_history: Optional[Dict[Any, Deque]] = None,
    ) -> List[dict]:
        """Attach position / distance / motion to each detection (mutates copies)."""
        frame_area = max(float(frame_width * frame_height), 1.0)
        enriched: List[dict] = []

        for det in detections:
            item = dict(det)
            cx, cy = item.get("center", [0, 0])
            area = float(item.get("area") or 0.0)
            name = str(item.get("class") or item.get("label") or "object")
            track_id = item.get("tracked_id")

            position = self._position(float(cx), frame_width)
            distance = self._distance(name, area, frame_area)

            history = self._resolve_history(track_id, track_history)
            sample = {
                "center": [float(cx), float(cy)],
                "area": area,
                "position": position,
                "distance": distance,
                "name": name,
            }
            history.append(sample)
            if track_id is not None:
                self._histories[track_id].append(sample)

            motion = self._infer_motion(history, frame_width)
            item.update(
                {
                    "name": name,
                    "position": position,
                    "distance": distance,
                    "motion": motion,
                    "area_ratio": round(area / frame_area, 4),
                }
            )
            enriched.append(item)

        return enriched

    def _resolve_history(
        self,
        track_id: Any,
        track_history: Optional[Dict[Any, Deque]],
    ) -> Deque[dict]:
        if track_id is not None and track_history and track_id in track_history:
            # Prefer detector's native track history when available
            native = track_history[track_id]
            converted: Deque[dict] = deque(maxlen=12)
            for entry in list(native)[-12:]:
                if isinstance(entry, dict) and "center" in entry:
                    converted.append(
                        {
                            "center": entry["center"],
                            "area": float(entry.get("area") or 0.0),
                            "name": entry.get("class", ""),
                        }
                    )
            if converted:
                return converted
        if track_id is not None:
            return self._histories[track_id]
        return deque(maxlen=12)

    def _position(self, cx: float, frame_width: int) -> Position:
        if frame_width <= 0:
            return "center"
        ratio = cx / float(frame_width)
        if ratio < self.left_max:
            return "left"
        if ratio > self.right_min:
            return "right"
        return "center"

    def _distance(self, name: str, area: float, frame_area: float) -> Distance:
        ratio = area / frame_area
        hint = self.CLASS_SIZE_HINT.get(name.lower(), 0.1)
        # Normalize relative to typical size of that class
        adjusted = ratio / max(hint, 0.01)

        near_cut = self.near_area_ratio / max(hint, 0.01) * hint
        far_cut = self.far_area_ratio / max(hint, 0.01) * hint

        # Simpler absolute thresholds with mild class tempering
        if ratio >= self.near_area_ratio or adjusted >= 1.4:
            return "near"
        if ratio <= self.far_area_ratio or adjusted <= 0.35:
            return "far"
        # Bridge using tempered cuts
        if ratio >= near_cut * 0.85:
            return "near"
        if ratio <= far_cut * 1.2:
            return "far"
        return "medium"

    def _infer_motion(self, history: Deque[dict], frame_width: int) -> Motion:
        if len(history) < 3:
            return "stationary"

        recent = list(history)[-6:]
        first = recent[0]
        last = recent[-1]
        dx = float(last["center"][0]) - float(first["center"][0])
        dy = float(last["center"][1]) - float(first["center"][1])
        dist = (dx * dx + dy * dy) ** 0.5

        areas = [float(h.get("area") or 0.0) for h in recent if h.get("area")]
        area_start = areas[0] if areas else 0.0
        area_end = areas[-1] if areas else 0.0
        area_ratio = (area_end / area_start) if area_start > 1 else 1.0

        # Approaching / moving away dominate when size changes clearly
        if area_ratio >= self.approach_area_growth and area_end > area_start:
            return "approaching"
        if area_ratio <= self.retreat_area_shrink and area_end < area_start:
            return "moving_away"

        # Crossing path: significant horizontal travel across center band
        if abs(dx) >= self.cross_min_dx_ratio * max(frame_width, 1):
            mid_x = frame_width / 2.0
            crossed = (first["center"][0] - mid_x) * (last["center"][0] - mid_x) < 0
            entered_center = abs(last["center"][0] - mid_x) < frame_width * 0.2
            if crossed or entered_center:
                return "crossing_path"

        if dist >= self.motion_pixel_threshold:
            return "moving"
        return "stationary"

    def clear(self) -> None:
        self._histories.clear()
