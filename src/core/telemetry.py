"""Telemetry helpers for reporting worker metrics."""
from __future__ import annotations

import socket
import time
from typing import Any, Dict, Optional

import httpx


class TelemetryError(RuntimeError):
    """Raised when telemetry emission fails after retries."""


class TelemetryClient:
    """Thin wrapper around httpx for structured telemetry events."""

    def __init__(
        self,
        base_url: Optional[str],
        *,
        cert: Optional[tuple[str, str]] = None,
        verify: Optional[str | bool] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self._client = client or httpx.Client(timeout=timeout, verify=verify, cert=cert) if base_url else None

    def emit(self, event: str, payload: Dict[str, Any]) -> Optional[httpx.Response]:
        if not self.base_url or not self._client:
            return None
        data = {
            "event": event,
            "timestamp": time.time(),
            "hostname": socket.gethostname(),
            "payload": payload,
        }
        response = self._client.post(self.base_url, json=data)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - simple pass-through
            raise TelemetryError(str(exc)) from exc
        return response

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class HeartbeatSender:
    """Periodically send heartbeat payloads to a telemetry endpoint."""

    def __init__(self, client: TelemetryClient, endpoint: Optional[str] = None) -> None:
        self.client = client
        self.endpoint = endpoint.rstrip("/") if endpoint else None

    def send(self, worker_id: str, payload: Dict[str, Any]) -> Optional[httpx.Response]:
        if self.endpoint is None:
            return self.client.emit("heartbeat", payload)
        data = {
            "worker_id": worker_id,
            "timestamp": time.time(),
            "payload": payload,
        }
        return self.client.emit(self.endpoint, data) if isinstance(self.client, TelemetryClient) else None


__all__ = ["TelemetryClient", "TelemetryError", "HeartbeatSender"]
