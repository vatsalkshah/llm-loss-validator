"""Worker level telemetry utilities."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..core.telemetry import TelemetryClient


class WorkerTelemetry:
    """Compose high-level telemetry payloads for workers."""

    def __init__(self, client: Optional[TelemetryClient], worker_id: str) -> None:
        self._client = client
        self.worker_id = worker_id

    def emit_status(self, *, mode: str, cache: Dict[str, Any], assignment_id: Optional[str] = None) -> None:
        if self._client is None:
            return
        payload = {
            "worker_id": self.worker_id,
            "mode": mode,
            "timestamp": time.time(),
            "cache": cache,
            "assignment_id": assignment_id,
        }
        self._client.emit("worker_status", payload)


__all__ = ["WorkerTelemetry"]
