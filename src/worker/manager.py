"""Dual-mode worker supervisor."""
from __future__ import annotations

import threading
from typing import Callable, Optional

from .state import WorkerMode, WorkerState
from .telemetry import WorkerTelemetry
from .validation_poller import ValidationPoller


class DualModeWorker:
    """Coordinate validation polling and inference supervision."""

    def __init__(
        self,
        *,
        mode: WorkerMode,
        validation_runner,
        validation_task_id: Optional[str] = None,
        fed_ledger=None,
        inference_server: Optional[Callable[[], None]] = None,
        telemetry: Optional[WorkerTelemetry] = None,
    ) -> None:
        self.state = WorkerState(mode=mode)
        self._validation_runner = validation_runner
        self._telemetry = telemetry
        self._fed_ledger = fed_ledger
        self._inference_server = inference_server
        self._task_id = validation_task_id
        self._poller = ValidationPoller(fed_ledger, validation_task_id, interval=5.0) if validation_task_id else None
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.state.mode in {WorkerMode.VALIDATION, WorkerMode.DUAL} and self._poller is not None:
            t = threading.Thread(target=self._validation_loop, name="validation-loop", daemon=True)
            t.start()
            self._threads.append(t)

        if self.state.mode in {WorkerMode.INFERENCE, WorkerMode.DUAL} and self._inference_server is not None:
            t = threading.Thread(target=self._run_inference, name="inference-server", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)

    # ------------------------------------------------------------------
    def _validation_loop(self) -> None:
        self.state.validation_active = True
        while not self._stop_event.is_set():
            payload = self._poller.poll_once() if self._poller else None
            if not payload:
                self._stop_event.wait(1.0)
                continue
            assignment_id = payload.get("id")
            self.state.last_assignment_id = assignment_id
            if hasattr(self._validation_runner, "run"):
                from ..core.validation_runner import AssignmentContext

                context = AssignmentContext(
                    task_id=self._task_id or "unknown",
                    assignment_id=assignment_id,
                    payload=payload,
                )
                self._validation_runner.run(context)
            else:
                self._validation_runner(payload)
            if self._telemetry is not None:
                self._telemetry.emit_status(mode=self.state.mode.value, cache={}, assignment_id=assignment_id)
        self.state.validation_active = False

    def _run_inference(self) -> None:
        self.state.inference_active = True
        try:
            if self._inference_server is not None:
                self._inference_server()
        finally:
            self.state.inference_active = False


__all__ = ["DualModeWorker"]
