"""
Real-time safety / hazard engine for accessibility-focused vision.
Consumes enriched detections (position, distance, motion) — never raw frames.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Priority ranks (lower number = higher urgency for voice)
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3

VEHICLES = {"car", "motorcycle", "bicycle", "bus", "truck"}
OBSTACLE_FURNITURE = {
    "chair",
    "couch",
    "dining table",
    "bed",
    "potted plant",
    "bench",
    "toilet",
    "tv",
    "refrigerator",
    "oven",
    "sink",
    "microwave",
}
PATH_OBJECTS = VEHICLES | OBSTACLE_FURNITURE | {
    "person",
    "backpack",
    "suitcase",
    "handbag",
    "dog",
    "cat",
    "fire hydrant",
    "stop sign",
    "parking meter",
}

# COCO has no stairs/door by default — reserved for future dedicated models
FUTURE_HAZARDS = {"stairs", "steps", "stair", "door", "entrance", "exit"}


class SafetyEngine:
    """Classify hazards, path status, and object priority from live scene state."""

    def __init__(
        self,
        path_band_top: float = 0.45,
        path_band_bottom: float = 1.0,
        center_left: float = 0.33,
        center_right: float = 0.66,
        near_blocks: bool = True,
    ) -> None:
        self.path_band_top = path_band_top
        self.path_band_bottom = path_band_bottom
        self.center_left = center_left
        self.center_right = center_right
        self.near_blocks = near_blocks

    def analyze(
        self,
        objects: List[dict],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        hazards: List[dict] = []
        prioritized: List[dict] = []

        for obj in objects:
            ranked = self._rank_object(obj)
            prioritized.append(ranked)
            if ranked["priority"] <= PRIORITY_HIGH:
                hazards.append(ranked)

        path = self._assess_path(objects, frame_width, frame_height)
        danger_level = self._danger_level(hazards, path)

        return {
            "objects": prioritized,
            "hazards": hazards,
            "path": path,
            "danger": danger_level,
            "hazard": hazards[0] if hazards else None,
        }

    def _rank_object(self, obj: dict) -> dict:
        name = str(obj.get("name") or obj.get("class") or "object").lower()
        position = obj.get("position", "center")
        distance = obj.get("distance", "medium")
        motion = obj.get("motion", "stationary")
        confidence = float(obj.get("confidence") or 0.0)
        in_path = position == "center" and distance in ("near", "medium")

        priority = PRIORITY_LOW
        kind = "info"
        reason = "detected"

        if name in FUTURE_HAZARDS and ("stair" in name or name == "steps"):
            priority = PRIORITY_CRITICAL
            kind = "stairs"
            reason = "stairs_ahead"
        elif name in FUTURE_HAZARDS and name in ("door", "entrance", "exit"):
            priority = PRIORITY_MEDIUM
            kind = "door"
            reason = "door_nearby"
        elif name in VEHICLES:
            if motion == "approaching" or (distance == "near" and position == "center"):
                priority = PRIORITY_CRITICAL
                kind = "vehicle"
                reason = "vehicle_approaching" if motion == "approaching" else "vehicle_near"
            elif distance in ("near", "medium") or motion in ("moving", "crossing_path"):
                priority = PRIORITY_HIGH
                kind = "vehicle"
                reason = "vehicle_nearby"
            else:
                priority = PRIORITY_MEDIUM
                kind = "vehicle"
                reason = "vehicle_visible"
        elif name == "person":
            if motion == "approaching" and distance in ("near", "medium"):
                priority = PRIORITY_HIGH
                kind = "person"
                reason = "person_approaching"
            elif motion == "crossing_path" or (in_path and distance == "near"):
                priority = PRIORITY_HIGH
                kind = "person"
                reason = "person_in_path"
            elif distance == "near" or position == "center":
                priority = PRIORITY_MEDIUM
                kind = "person"
                reason = "person_nearby"
            else:
                priority = PRIORITY_LOW
                kind = "person"
                reason = "person_visible"
        elif name in OBSTACLE_FURNITURE or name in PATH_OBJECTS:
            if in_path and distance == "near":
                priority = PRIORITY_CRITICAL
                kind = "obstacle"
                reason = "obstacle_in_path"
            elif in_path:
                priority = PRIORITY_HIGH
                kind = "obstacle"
                reason = "path_obstacle"
            elif distance == "near":
                priority = PRIORITY_MEDIUM
                kind = "obstacle"
                reason = "obstacle_near"
            else:
                priority = PRIORITY_LOW
                kind = "object"
                reason = "object_visible"
        else:
            if distance == "near" and position == "center":
                priority = PRIORITY_MEDIUM
                kind = "object"
                reason = "object_ahead"
            else:
                priority = PRIORITY_LOW
                kind = "object"
                reason = "object_visible"

        # Soften low-confidence claims
        if confidence < 0.45 and priority <= PRIORITY_HIGH:
            priority = min(priority + 1, PRIORITY_LOW)

        out = dict(obj)
        out.update(
            {
                "name": name,
                "priority": priority,
                "kind": kind,
                "reason": reason,
                "in_path": bool(in_path),
            }
        )
        return out

    def _assess_path(
        self,
        objects: List[dict],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """Lower/central band path awareness: clear / partially_blocked / blocked."""
        lanes = {"left": [], "center": [], "right": []}

        for obj in objects:
            name = str(obj.get("name") or obj.get("class") or "").lower()
            if name not in PATH_OBJECTS and name not in FUTURE_HAZARDS:
                # Still consider large near center boxes as potential blockers
                if obj.get("distance") != "near" or obj.get("position") != "center":
                    continue

            bbox = obj.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = bbox
            cy = float(obj.get("center", [(x1 + x2) / 2, (y1 + y2) / 2])[1])
            y_ratio = cy / max(frame_height, 1)

            if y_ratio < self.path_band_top or y_ratio > self.path_band_bottom:
                continue

            # Prefer objects that intersect lower path band
            band_top = int(frame_height * self.path_band_top)
            if y2 < band_top:
                continue

            distance = obj.get("distance", "medium")
            if distance == "far":
                continue

            position = obj.get("position", "center")
            if position not in lanes:
                position = "center"
            lanes[position].append(obj)

        def lane_blocked(items: List[dict]) -> bool:
            if not items:
                return False
            return any(
                i.get("distance") == "near"
                or (i.get("distance") == "medium" and i.get("position") == "center")
                for i in items
            )

        left_b = lane_blocked(lanes["left"])
        center_b = lane_blocked(lanes["center"])
        right_b = lane_blocked(lanes["right"])

        blocked_count = sum([left_b, center_b, right_b])
        status = "clear"
        if center_b and (left_b or right_b):
            status = "blocked"
        elif center_b or blocked_count >= 2:
            status = "partially_blocked" if blocked_count < 3 else "blocked"
        elif left_b or right_b:
            status = "partially_blocked"

        blocking = None
        if lanes["center"]:
            blocking = max(
                lanes["center"],
                key=lambda o: float(o.get("area") or 0.0),
            )
        elif lanes["left"] or lanes["right"]:
            pool = lanes["left"] + lanes["right"]
            blocking = max(pool, key=lambda o: float(o.get("area") or 0.0))

        suggestion = None
        if center_b and not right_b and left_b:
            suggestion = "right"
        elif center_b and not left_b and right_b:
            suggestion = "left"
        elif center_b and not left_b and not right_b:
            # Prefer clearer side by emptiness
            suggestion = "right" if not right_b else "left"
        elif center_b and left_b and not right_b:
            suggestion = "right"
        elif center_b and right_b and not left_b:
            suggestion = "left"

        return {
            "status": status,
            "lanes": {
                "left": "blocked" if left_b else "clear",
                "center": "blocked" if center_b else "clear",
                "right": "blocked" if right_b else "clear",
            },
            "blocking_object": (
                str(blocking.get("name") or blocking.get("class"))
                if blocking
                else None
            ),
            "position": blocking.get("position") if blocking else None,
            "suggestion": suggestion,
        }

    def _danger_level(self, hazards: List[dict], path: Dict[str, Any]) -> str:
        if any(h.get("priority") == PRIORITY_CRITICAL for h in hazards):
            return "critical"
        if path.get("status") == "blocked":
            return "high"
        if any(h.get("priority") == PRIORITY_HIGH for h in hazards):
            return "high"
        if path.get("status") == "partially_blocked":
            return "medium"
        if hazards:
            return "medium"
        return "low"
