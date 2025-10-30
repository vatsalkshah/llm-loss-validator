"""Worker telemetry integration built on the core heartbeat reporter."""

from __future__ import annotations

from typing import Optional

from client.fed_ledger import FedLedger
from core.model_cache import ModelCacheManager
from core.telemetry import (
    HeartbeatReporter,
    TelemetryConfig,
    TelemetryPublisher,
)
from worker.state import WorkerState


class TelemetryReporter:
    """High-level telemetry facade used by the dual-mode worker."""

    def __init__(
        self,
        *,
        state: WorkerState,
        cache_manager: ModelCacheManager,
        fed_ledger: Optional[FedLedger] = None,
        interval: Optional[float] = None,
        webhook_url: Optional[str] = None,
        location: Optional[str] = None,
        worker_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._config = TelemetryConfig.from_env(
            interval_seconds=interval,
            webhook_url=webhook_url,
            location=location,
            worker_id=worker_id,
            enabled=enabled,
        )

        publisher = TelemetryPublisher(
            fed_ledger=fed_ledger,
            webhook_url=self._config.webhook_url,
        )

        self._reporter = HeartbeatReporter(
            state=state,
            cache_manager=cache_manager,
            config=self._config,
            publisher=publisher,
        )

    async def start(self) -> None:
        await self._reporter.start()

    async def stop(self) -> None:
        await self._reporter.stop()

    async def collect_snapshot(self) -> dict:
        snapshot = await self._reporter.collect_snapshot()
        return snapshot.as_dict()

    def get_latest_snapshot(self) -> Optional[dict]:
        return self._reporter.get_latest_snapshot()
