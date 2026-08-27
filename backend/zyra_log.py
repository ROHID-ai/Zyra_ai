"""
Lightweight tagged logging for Zyra backend.
Avoids per-frame spam unless debug_mode is enabled in config.
"""
from __future__ import annotations

import time
from typing import Optional

_debug_mode = False
_last_throttle: dict[str, float] = {}


def set_debug(enabled: bool) -> None:
    global _debug_mode
    _debug_mode = enabled


def log(tag: str, message: str, *, throttle_key: Optional[str] = None, throttle_s: float = 2.0) -> None:
    """Print a tagged log line. Optional throttle for repeated messages."""
    if throttle_key:
        now = time.time()
        last = _last_throttle.get(throttle_key, 0.0)
        if now - last < throttle_s:
            return
        _last_throttle[throttle_key] = now

    print(f"[{tag.upper()}] {message}")


def debug(tag: str, message: str) -> None:
    if _debug_mode:
        print(f"[DEBUG:{tag.upper()}] {message}")
