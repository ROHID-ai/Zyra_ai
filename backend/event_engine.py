"""
Event engine: emit meaningful state-change events for voice (not per-frame spam).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from safety_engine import PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM


class EventEngine:
    """Track scene deltas and produce prioritized speakable events."""

    def __init__(
        self,
        event_cooldown: float = 4.0,
        critical_cooldown: float = 2.0,
        path_cooldown: float = 5.0,
        max_recent: int = 20,
    ) -> None:
        self.event_cooldown = event_cooldown
        self.critical_cooldown = critical_cooldown
        self.path_cooldown = path_cooldown
        self.max_recent = max_recent

        self._last_spoken: Dict[str, float] = {}
        self._known_tracks: Dict[Any, dict] = {}
        self._last_path_status: Optional[str] = None
        self._last_currency_sig: Optional[str] = None
        self.recent_events: List[dict] = []
        self.last_spoken_text: str = ""

    def process_scene(
        self,
        safety: Dict[str, Any],
        currency_summary: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Return newly actionable events for this frame's logical state."""
        now = time.time()
        events: List[dict] = []
        objects = safety.get("objects") or []
        path = safety.get("path") or {}

        # Track-level object events
        seen_ids = set()
        for obj in objects:
            track_id = obj.get("tracked_id")
            key = track_id if track_id is not None else f"{obj.get('name')}_{obj.get('position')}"
            seen_ids.add(key)
            prev = self._known_tracks.get(key)
            events.extend(self._object_events(obj, prev, now))
            self._known_tracks[key] = {
                "name": obj.get("name"),
                "position": obj.get("position"),
                "distance": obj.get("distance"),
                "motion": obj.get("motion"),
                "reason": obj.get("reason"),
                "priority": obj.get("priority"),
                "in_path": obj.get("in_path"),
                "kind": obj.get("kind"),
            }

        # Drop stale tracks
        stale = [k for k in self._known_tracks if k not in seen_ids]
        for k in stale:
            del self._known_tracks[k]

        # Path transitions
        path_status = path.get("status", "clear")
        if path_status != self._last_path_status:
            events.extend(self._path_events(path, self._last_path_status, now))
            self._last_path_status = path_status

        # Currency aggregate change
        if currency_summary:
            events.extend(self._currency_events(currency_summary, now))

        # Deduplicate / cooldown filter
        filtered = [e for e in events if self._allow(e, now)]
        filtered.sort(key=lambda e: (e.get("priority", 9), e.get("ts", now)))

        for event in filtered:
            self._record(event)

        return filtered

    def _object_events(
        self, obj: dict, prev: Optional[dict], now: float
    ) -> List[dict]:
        name = str(obj.get("name") or "object")
        position = obj.get("position", "center")
        distance = obj.get("distance", "medium")
        motion = obj.get("motion", "stationary")
        priority = int(obj.get("priority", PRIORITY_MEDIUM))
        kind = obj.get("kind", "object")
        reason = obj.get("reason", "detected")
        track_id = obj.get("tracked_id")
        events: List[dict] = []

        def make(
            etype: str,
            message: str,
            prio: int,
            critical: bool = False,
        ) -> dict:
            return {
                "type": etype,
                "message": message,
                "priority": prio,
                "critical": critical,
                "object": name,
                "position": position,
                "distance": distance,
                "motion": motion,
                "kind": kind,
                "track_id": track_id,
                "ts": now,
                "key": f"{etype}:{track_id or name}:{position}",
            }

        if prev is None:
            # First meaningful appearance — only announce useful objects
            if kind == "vehicle":
                if motion == "approaching":
                    events.append(
                        make(
                            "VEHICLE_APPROACHING",
                            f"Warning. Vehicle approaching from your {position}.",
                            PRIORITY_CRITICAL,
                            True,
                        )
                    )
                elif distance != "far":
                    side = f" on your {position}" if position != "center" else " ahead"
                    events.append(
                        make(
                            "OBJECT_ENTERED",
                            f"{name.capitalize()}{side}.",
                            PRIORITY_HIGH,
                        )
                    )
            elif kind == "stairs":
                events.append(
                    make(
                        "STAIRS_DETECTED",
                        "Stairs ahead.",
                        PRIORITY_CRITICAL,
                        True,
                    )
                )
            elif kind == "door":
                side = f" on your {position}" if position != "center" else " ahead"
                events.append(
                    make(
                        "OBJECT_ENTERED",
                        f"Door{side}.",
                        PRIORITY_MEDIUM,
                    )
                )
            elif kind == "person":
                if position == "center" or distance == "near":
                    events.append(
                        make("OBJECT_ENTERED", "Person ahead.", PRIORITY_MEDIUM)
                    )
                else:
                    events.append(
                        make(
                            "OBJECT_ENTERED",
                            f"Person on your {position}.",
                            PRIORITY_MEDIUM,
                        )
                    )
            elif kind == "obstacle" and (obj.get("in_path") or distance == "near"):
                if position == "center":
                    events.append(
                        make(
                            "OBJECT_ENTERED_PATH",
                            "Obstacle directly ahead.",
                            PRIORITY_CRITICAL if distance == "near" else PRIORITY_HIGH,
                            distance == "near",
                        )
                    )
                else:
                    events.append(
                        make(
                            "OBJECT_ENTERED",
                            f"{name.capitalize()} on your {position}.",
                            PRIORITY_MEDIUM,
                        )
                    )
            elif priority <= PRIORITY_MEDIUM and distance != "far":
                # Useful but not chatty for every bottle far away
                if position == "center":
                    events.append(
                        make(
                            "OBJECT_ENTERED",
                            f"{name.capitalize()} ahead.",
                            priority,
                        )
                    )
            return events

        # State changes for known tracks
        if prev.get("motion") != "approaching" and motion == "approaching":
            if kind == "vehicle":
                events.append(
                    make(
                        "VEHICLE_APPROACHING",
                        f"Warning. Vehicle approaching from your {position}.",
                        PRIORITY_CRITICAL,
                        True,
                    )
                )
            elif kind == "person":
                events.append(
                    make(
                        "OBJECT_APPROACHING",
                        "Person approaching.",
                        PRIORITY_HIGH,
                    )
                )
            else:
                events.append(
                    make(
                        "OBJECT_APPROACHING",
                        f"{name.capitalize()} approaching.",
                        PRIORITY_HIGH,
                    )
                )

        if prev.get("motion") != "crossing_path" and motion == "crossing_path":
            if kind == "person":
                events.append(
                    make(
                        "OBJECT_MOVED",
                        "Person moving across your path.",
                        PRIORITY_HIGH,
                    )
                )
            elif kind == "vehicle":
                events.append(
                    make(
                        "OBJECT_MOVED",
                        f"Warning. {name.capitalize()} crossing your path.",
                        PRIORITY_CRITICAL,
                        True,
                    )
                )

        if prev.get("distance") != "near" and distance == "near":
            if kind == "vehicle":
                events.append(
                    make(
                        "OBJECT_BECAME_NEAR",
                        f"Warning. {name.capitalize()} close on your {position}.",
                        PRIORITY_CRITICAL,
                        True,
                    )
                )
            elif kind == "person":
                events.append(
                    make(
                        "OBJECT_BECAME_NEAR",
                        f"Person close on your {position}.",
                        PRIORITY_HIGH,
                    )
                )
            elif obj.get("in_path") or position == "center":
                events.append(
                    make(
                        "OBJECT_BECAME_NEAR",
                        f"{name.capitalize()} close ahead.",
                        PRIORITY_HIGH,
                    )
                )

        was_in_path = bool(prev.get("in_path"))
        now_in_path = bool(obj.get("in_path"))
        if not was_in_path and now_in_path:
            if kind == "person":
                events.append(
                    make(
                        "OBJECT_ENTERED_PATH",
                        "Person entering your path.",
                        PRIORITY_HIGH,
                    )
                )
            else:
                events.append(
                    make(
                        "OBJECT_ENTERED_PATH",
                        "Obstacle directly ahead.",
                        PRIORITY_HIGH,
                    )
                )
        if was_in_path and not now_in_path:
            events.append(
                make(
                    "OBJECT_LEFT_PATH",
                    f"{name.capitalize()} left your path.",
                    PRIORITY_MEDIUM,
                )
            )

        if prev.get("position") != position and priority <= PRIORITY_HIGH:
            events.append(
                make(
                    "OBJECT_MOVED",
                    f"{name.capitalize()} now on your {position}.",
                    priority,
                )
            )

        return events

    def _path_events(
        self, path: dict, prev_status: Optional[str], now: float
    ) -> List[dict]:
        status = path.get("status", "clear")
        blocking = path.get("blocking_object")
        suggestion = path.get("suggestion")
        events: List[dict] = []

        if status == "blocked" and prev_status != "blocked":
            msg = "Path ahead appears blocked."
            if suggestion:
                msg += f" Move slightly {suggestion}."
            elif blocking:
                msg = f"The center path appears blocked by a {blocking}."
            events.append(
                {
                    "type": "PATH_BLOCKED",
                    "message": msg,
                    "priority": PRIORITY_CRITICAL,
                    "critical": True,
                    "ts": now,
                    "key": "PATH_BLOCKED",
                }
            )
        elif status == "partially_blocked" and prev_status in (None, "clear"):
            msg = "There appears to be an obstacle ahead."
            if suggestion:
                msg += f" The {suggestion} may be clearer."
            events.append(
                {
                    "type": "PATH_BLOCKED",
                    "message": msg,
                    "priority": PRIORITY_HIGH,
                    "critical": False,
                    "ts": now,
                    "key": "PATH_PARTIAL",
                }
            )
        elif status == "clear" and prev_status in ("blocked", "partially_blocked"):
            events.append(
                {
                    "type": "PATH_CLEARED",
                    "message": "The path appears clear.",
                    "priority": PRIORITY_MEDIUM,
                    "critical": False,
                    "ts": now,
                    "key": "PATH_CLEARED",
                }
            )
        return events

    def _currency_events(self, summary: Dict[str, Any], now: float) -> List[dict]:
        sig = summary.get("signature")
        if not sig or sig == self._last_currency_sig:
            return []
        self._last_currency_sig = sig
        message = summary.get("spoken") or "Currency detected."
        return [
            {
                "type": "CURRENCY_CHANGED",
                "message": message,
                "priority": PRIORITY_MEDIUM,
                "critical": False,
                "ts": now,
                "key": f"CURRENCY:{sig}",
            }
        ]

    def _allow(self, event: dict, now: float) -> bool:
        key = event.get("key") or event.get("type")
        cooldown = (
            self.critical_cooldown
            if event.get("critical")
            else self.path_cooldown
            if str(event.get("type", "")).startswith("PATH")
            else self.event_cooldown
        )
        last = self._last_spoken.get(key, 0.0)
        if now - last < cooldown:
            return False
        self._last_spoken[key] = now
        return True

    def _record(self, event: dict) -> None:
        self.last_spoken_text = event.get("message", "")
        self.recent_events.append(
            {
                "type": event.get("type"),
                "message": event.get("message"),
                "priority": event.get("priority"),
                "critical": bool(event.get("critical")),
                "ts": event.get("ts"),
            }
        )
        if len(self.recent_events) > self.max_recent:
            self.recent_events = self.recent_events[-self.max_recent :]

    def reset(self) -> None:
        self._known_tracks.clear()
        self._last_path_status = None
        self._last_currency_sig = None
