"""Concurrency helpers for coordinating inference and validation."""
from __future__ import annotations

import threading
from contextlib import contextmanager


class InferenceActivityTracker:
    """Track in-flight inference calls and allow validation to block them."""

    def __init__(self) -> None:
        self._active = 0
        self._lock = threading.Lock()
        self._can_start = threading.Event()
        self._can_start.set()

    @contextmanager
    def track(self):
        self._can_start.wait()
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                if self._active == 0:
                    self._can_start.set()

    def pause(self) -> None:
        """Block new inference calls until `resume` is invoked."""

        self._can_start.clear()
        with self._lock:
            if self._active == 0:
                self._can_start.set()

    def resume(self) -> None:
        self._can_start.set()

    def active_requests(self) -> int:
        with self._lock:
            return self._active


__all__ = ["InferenceActivityTracker"]
