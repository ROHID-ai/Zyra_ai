"""
Groq intelligence layer — language / Q&A over structured LIVE detection state.
Never receives raw video frames. Never blocks critical safety TTS.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class GroqService:
    """Optional Groq client for natural language over structured scene context."""

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        self.model = model
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self._client = None
        self._lock = threading.Lock()
        self.enabled = False
        self.last_error: Optional[str] = None

        if not self.api_key:
            print("[GroqService] GROQ_API_KEY not set — local responses only")
            return

        try:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
            self.enabled = True
            print("[GroqService] Ready")
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[GroqService] Disabled ({exc})")

    def answer_question(self, question: str, live_state: Dict[str, Any]) -> str:
        """Answer using current live structured state (not a new camera capture)."""
        local = self._local_answer(question, live_state)
        if not self.enabled or not self._client:
            return local

        try:
            system = (
                "You are Zyra AI, a real-time vision assistant for blind and "
                "visually impaired users. Answer ONLY from the provided JSON "
                "live detection state. Be concise (1-2 short sentences). "
                "Use conservative language: say 'appears' / 'seems'. "
                "Never invent objects. Never give exact meters. "
                "Never say 'safe to walk'. Prefer 'the path appears clear'."
            )
            user = (
                f"User question: {question}\n\n"
                f"Live detection state JSON:\n{json.dumps(live_state, default=str)[:3500]}"
            )
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or local
        except Exception as exc:
            self.last_error = str(exc)
            return local

    def polish_noncritical(self, message: str, live_state: Dict[str, Any]) -> str:
        """Optional wording polish for non-critical events. Fail open to original."""
        if not self.enabled or not self._client or not message:
            return message
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite accessibility voice prompts to be short, clear, "
                            "and conservative. Keep meaning. Max 18 words. "
                            "Return only the rewritten sentence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Original: {message}\nContext: {json.dumps(live_state, default=str)[:800]}",
                    },
                ],
                temperature=0.2,
                max_tokens=60,
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or message
        except Exception as exc:
            self.last_error = str(exc)
            return message

    def classify_intent(self, utterance: str) -> Dict[str, str]:
        """Map natural speech to an action. Local-first; Groq optional assist."""
        local = self._local_intent(utterance)
        if local["action"] != "unknown" or not self.enabled or not self._client:
            return local
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the user utterance for Zyra vision assistant. "
                            "Return JSON only with keys action and query. "
                            "action one of: start_object, start_currency, stop, ask, help, unknown. "
                            "If action is ask, put the cleaned question in query."
                        ),
                    },
                    {"role": "user", "content": utterance},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            raw = (completion.choices[0].message.content or "").strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(raw[start : end + 1])
                action = str(data.get("action") or "unknown")
                query = str(data.get("query") or utterance)
                return {"action": action, "query": query}
        except Exception as exc:
            self.last_error = str(exc)
        return local

    def _local_intent(self, utterance: str) -> Dict[str, str]:
        cmd = utterance.lower().strip()
        ask_hints = (
            "what do you see",
            "what's ahead",
            "whats ahead",
            "what is ahead",
            "is there anyone",
            "where is",
            "where's",
            "is the path",
            "what's on my",
            "whats on my",
            "read the scene",
            "describe",
            "how many",
            "anyone near",
        )
        if any(h in cmd for h in ("what just happened", "what happened", "recent")):
            return {"action": "recent_events", "query": utterance}
        if "read text" in cmd or "read the text" in cmd or cmd == "read":
            return {"action": "read_text", "query": utterance}
        if any(h in cmd for h in ask_hints) or cmd.endswith("?"):
            return {"action": "ask", "query": utterance}
        if any(x in cmd for x in ("start vision", "start object", "object mode", "activate", "start")):
            return {"action": "start_object", "query": ""}
        if any(x in cmd for x in ("currency", "money", "rupee", "check currency", "notes")):
            if "check" in cmd or "currency" in cmd or "money" in cmd or "rupee" in cmd:
                return {"action": "start_currency", "query": ""}
        if any(x in cmd for x in ("stop vision", "stop", "deactivate", "quit")):
            return {"action": "stop", "query": ""}
        if "help" in cmd:
            return {"action": "help", "query": ""}
        if "object" in cmd:
            return {"action": "start_object", "query": ""}
        return {"action": "unknown", "query": utterance}

    def _local_answer(self, question: str, live_state: Dict[str, Any]) -> str:
        q = question.lower()
        objects = live_state.get("objects") or []
        path = live_state.get("path") or {}
        currency = live_state.get("currency")
        mode = live_state.get("mode")

        if "currency" in q or "rupee" in q or "money" in q or "note" in q:
            if currency and currency.get("spoken"):
                return currency["spoken"]
            return "I do not currently see any currency notes."

        if any(
            x in q
            for x in ("just happened", "what happened", "recent events", "recently")
        ):
            recent = live_state.get("recent_events") or []
            messages = [
                e.get("message", "")
                for e in reversed(recent[-5:])
                if e.get("message")
            ]
            if not messages:
                return "Nothing notable has happened recently."
            return "Recently: " + ". ".join(messages) + "."

        if "path" in q or "blocked" in q or "clear" in q or "ahead" in q and "who" not in q:
            status = path.get("status", "clear") if isinstance(path, dict) else path
            if status == "blocked":
                blocking = path.get("blocking_object") if isinstance(path, dict) else None
                extra = f" by a {blocking}" if blocking else ""
                return f"The center path appears blocked{extra}."
            if status == "partially_blocked":
                suggestion = path.get("suggestion") if isinstance(path, dict) else None
                msg = "There appears to be an obstacle ahead."
                if suggestion:
                    msg += f" Moving slightly {suggestion} may be clearer."
                return msg
            if "ahead" in q and objects:
                ahead = [
                    o
                    for o in objects
                    if o.get("position") == "center" or o.get("in_path")
                ]
                if ahead:
                    names = ", ".join(sorted({o.get("name", "object") for o in ahead[:4]}))
                    return f"Ahead I can see: {names}."
            return "The path appears clear."

        for side in ("left", "right", "center"):
            if side in q or (side == "center" and "ahead" in q):
                side_objs = [o for o in objects if o.get("position") == side]
                if not side_objs:
                    label = "ahead" if side == "center" else f"on your {side}"
                    return f"I do not currently see anything clear {label}."
                bits = []
                for o in side_objs[:4]:
                    bits.append(
                        f"{o.get('name')} ({o.get('distance', 'medium')})"
                    )
                label = "ahead" if side == "center" else f"on your {side}"
                return f"{label.capitalize()}: " + ", ".join(bits) + "."

        if "where" in q:
            # Find mentioned object name
            for o in objects:
                name = str(o.get("name") or "")
                if name and name in q:
                    pos = o.get("position", "center")
                    dist = o.get("distance", "medium")
                    place = "ahead" if pos == "center" else f"on your {pos}"
                    return f"Your {name} appears to be {place}, relatively {dist}."
            # Common aliases
            aliases = {
                "phone": "cell phone",
                "mobile": "cell phone",
                "bag": "backpack",
                "table": "dining table",
            }
            for alias, canonical in aliases.items():
                if alias in q:
                    match = next((o for o in objects if o.get("name") == canonical), None)
                    if match:
                        pos = match.get("position", "center")
                        place = "ahead" if pos == "center" else f"on your {pos}"
                        return f"Your {canonical} appears to be {place}."
                    return f"I do not currently see a {alias}."

        if any(x in q for x in ("anyone", "person", "people", "someone")):
            people = [o for o in objects if o.get("name") == "person"]
            if not people:
                return "I do not currently see anyone nearby."
            bits = []
            for p in people[:3]:
                pos = p.get("position", "center")
                place = "ahead" if pos == "center" else f"on your {pos}"
                bits.append(f"{place}, {p.get('distance', 'medium')}")
            return "Person detected " + "; ".join(bits) + "."

        if any(x in q for x in ("see", "scene", "around", "detect", "there")):
            if mode == "currency" and currency and currency.get("spoken"):
                return currency["spoken"]
            if not mode:
                return "Vision is not active yet. Say start vision and I will describe your surroundings."
            if not objects:
                return "I am scanning, but I do not currently see clear objects."
            summary = []
            counts: Dict[str, int] = {}
            for o in objects:
                n = str(o.get("name") or "object")
                counts[n] = counts.get(n, 0) + 1
            for name, count in list(counts.items())[:6]:
                summary.append(f"{count} {name}" if count > 1 else name)
            path_status = path.get("status") if isinstance(path, dict) else path
            msg = "I can see " + ", ".join(summary) + "."
            if path_status and path_status != "clear":
                msg += f" Path appears {str(path_status).replace('_', ' ')}."
            return msg

        if not objects:
            return "I do not currently have clear detections to answer that."
        return self._local_answer("what do you see", live_state)

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model if self.enabled else None,
            "error": self.last_error or "",
        }
