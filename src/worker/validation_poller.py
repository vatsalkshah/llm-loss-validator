"""Polling helper for FedLedger assignments."""
from __future__ import annotations

import time
from typing import Optional


class ValidationPoller:
    def __init__(self, fed_ledger, task_id: str, *, interval: float = 60.0) -> None:
        self._fed_ledger = fed_ledger
        self._task_id = task_id
        self._interval = interval
        self._last_request = 0.0

    def poll_once(self) -> Optional[dict]:
        now = time.time()
        if now - self._last_request < self._interval:
            return None
        self._last_request = now
        response = self._fed_ledger.request_validation_assignment(self._task_id)
        if isinstance(response, dict):
            return response
        if response.status_code != 200:
            return None
        return response.json()


__all__ = ["ValidationPoller"]
