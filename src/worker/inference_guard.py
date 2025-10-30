"""
Inference activity tracking for dual-mode worker.

Provides an API for the inference server to signal active requests and
respect validation pre-emption.
"""

from __future__ import annotations

import contextlib
from typing import AsyncIterator

from worker.state import WorkerState


class InferenceBusyError(RuntimeError):
    """Raised when inference is temporarily unavailable due to validation."""


class InferenceActivityTracker:
    """Track active inference requests in coordination with the worker state."""

    def __init__(self, state: WorkerState):
        self._state = state

    async def acquire(self):
        """Attempt to acquire permission for an inference request."""
        if not await self._state.start_inference_request():
            raise InferenceBusyError("Inference unavailable while validation is pending")

    async def release(self):
        """Release an active inference request."""
        await self._state.end_inference_request()

    @contextlib.asynccontextmanager
    async def track(self) -> AsyncIterator[None]:
        """
        Async context manager to register an inference request lifecycle.

        Usage:
            async with tracker.track():
                ...
        """
        await self.acquire()
        try:
            yield
        finally:
            await self.release()

    @contextlib.asynccontextmanager
    async def __call__(self) -> AsyncIterator[None]:
        """Allow using tracker as an async context manager directly."""
        async with self.track():
            yield

    async def active_count(self) -> int:
        """Return the current number of active inference requests."""
        return await self._state.get_active_inference_count()
