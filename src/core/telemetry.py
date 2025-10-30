"""Background telemetry reporting utilities.

Provides reusable primitives for collecting worker health data and
publishing periodic heartbeat payloads to external services.
"""

from __future__ import annotations

import asyncio
import copy
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional, TYPE_CHECKING

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.gpu_utils import get_gpu_type

if TYPE_CHECKING:  # pragma: no cover - import guarded for type hints only
    from client.fed_ledger import FedLedger
    from core.model_cache import ModelCacheManager
    from worker.state import ValidationJob, WorkerState


telemetry_logger = logger.bind(component="telemetry")


def _env_flag(name: str, default: bool = True) -> bool:
    """Return a boolean environment flag."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TelemetryConfig:
    """Configuration for telemetry heartbeat publishing."""

    enabled: bool = True
    interval_seconds: float = 30.0
    webhook_url: Optional[str] = None
    location: Optional[str] = None
    worker_id: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        *,
        interval_seconds: Optional[float] = None,
        webhook_url: Optional[str] = None,
        location: Optional[str] = None,
        worker_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> "TelemetryConfig":
        """Create a telemetry configuration using environment defaults."""

        resolved_interval = interval_seconds
        if resolved_interval is None:
            env_interval = os.getenv("TELEMETRY_INTERVAL_SECONDS")
            if env_interval is not None:
                try:
                    resolved_interval = float(env_interval)
                except ValueError:
                    telemetry_logger.warning(
                        "Invalid TELEMETRY_INTERVAL_SECONDS value, falling back to default",
                        value=env_interval,
                    )
        if resolved_interval is None:
            resolved_interval = cls.interval_seconds

        resolved_enabled = enabled
        if resolved_enabled is None:
            resolved_enabled = _env_flag("TELEMETRY_ENABLED", default=True)

        return cls(
            enabled=resolved_enabled,
            interval_seconds=resolved_interval,
            webhook_url=webhook_url or os.getenv("TELEMETRY_WEBHOOK_URL"),
            location=location or os.getenv("TELEMETRY_LOCATION"),
            worker_id=worker_id
            or os.getenv("TELEMETRY_WORKER_ID")
            or os.getenv("WORKER_ID"),
        )


class NetworkStatsSampler:
    """Estimate network throughput via /proc/net/dev sampling."""

    def __init__(self) -> None:
        self._last_rx_bytes: Optional[int] = None
        self._last_tx_bytes: Optional[int] = None
        self._last_timestamp: Optional[float] = None

    def sample(self) -> Dict[str, float]:
        """Return upload/download speeds in Mbps based on byte deltas."""

        rx_bytes, tx_bytes = self._read_bytes()
        now = time.time()

        if self._last_timestamp is None:
            self._last_timestamp = now
            self._last_rx_bytes = rx_bytes
            self._last_tx_bytes = tx_bytes
            return {"download_mbps": 0.0, "upload_mbps": 0.0}

        elapsed = max(now - self._last_timestamp, 1e-6)
        download_bps = max(rx_bytes - self._last_rx_bytes, 0) / elapsed
        upload_bps = max(tx_bytes - self._last_tx_bytes, 0) / elapsed

        self._last_timestamp = now
        self._last_rx_bytes = rx_bytes
        self._last_tx_bytes = tx_bytes

        return {
            "download_mbps": round((download_bps * 8) / 1_000_000, 3),
            "upload_mbps": round((upload_bps * 8) / 1_000_000, 3),
        }

    def _read_bytes(self) -> tuple[int, int]:
        """Read aggregate RX/TX byte counters from /proc/net/dev."""

        total_rx = 0
        total_tx = 0

        try:
            with open("/proc/net/dev", "r", encoding="utf-8") as handle:
                lines = handle.readlines()[2:]
        except FileNotFoundError:
            telemetry_logger.warning("/proc/net/dev not available; network stats disabled")
            return 0, 0

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            iface, data = line.split(":", 1)
            interface = iface.strip()
            if interface == "lo":
                continue
            parts = data.split()
            if len(parts) < 16:
                continue
            try:
                rx_bytes = int(parts[0])
                tx_bytes = int(parts[8])
            except (ValueError, IndexError):
                continue
            total_rx += rx_bytes
            total_tx += tx_bytes

        return total_rx, total_tx


@dataclass
class TelemetrySnapshot:
    """Structured heartbeat payload."""

    timestamp: float
    cadence_seconds: float
    worker_id: Optional[str]
    location: Optional[str]
    worker_status: Dict[str, Any]
    cache: Dict[str, Any]
    gpu: Dict[str, Any]
    network: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "timestamp": self.timestamp,
            "cadence_seconds": self.cadence_seconds,
            "status": self.worker_status,
            "cache": self.cache,
            "gpu": self.gpu,
            "network": self.network,
        }

        if self.worker_id or self.location:
            payload["worker"] = {
                "id": self.worker_id,
                "location": self.location,
            }
        if self.metadata:
            payload["metadata"] = self.metadata

        return payload


@dataclass
class PublishResult:
    """Outcome of a telemetry publish attempt."""

    fed_ledger: Optional[bool] = None
    webhook: Optional[bool] = None

    def succeeded(self) -> bool:
        statuses = [status for status in (self.fed_ledger, self.webhook) if status is not None]
        return all(statuses) if statuses else True

    def as_dict(self) -> Dict[str, Optional[bool]]:
        return {"fed_ledger": self.fed_ledger, "webhook": self.webhook}


class TelemetryPublisher:
    """Publish telemetry payloads to configured sinks."""

    def __init__(
        self,
        *,
        fed_ledger: Optional["FedLedger"] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        self._fed_ledger = fed_ledger
        self._webhook_url = webhook_url

    async def publish(self, payload: Dict[str, Any]) -> PublishResult:
        result = PublishResult()

        if self._fed_ledger:
            result.fed_ledger = await asyncio.to_thread(self._publish_fed_ledger, payload)

        if self._webhook_url:
            result.webhook = await asyncio.to_thread(self._publish_webhook, payload)

        return result

    def _publish_fed_ledger(self, payload: Dict[str, Any]) -> bool:
        try:
            response = self._fed_ledger.report_worker_heartbeat(payload)  # type: ignore[arg-type]
            if response:
                telemetry_logger.debug("FedLedger heartbeat acknowledged")
            return bool(response)
        except Exception as exc:  # pragma: no cover - defensive logging
            telemetry_logger.error("FedLedger heartbeat failed", error=str(exc))
            return False

    def _publish_webhook(self, payload: Dict[str, Any]) -> bool:
        if not self._webhook_url:
            return True

        if not self._webhook_url.lower().startswith("https://"):
            telemetry_logger.error(
                "Telemetry webhook URL must use HTTPS",
                url=self._webhook_url,
            )
            return False

        try:
            self._post_with_retry(self._webhook_url, payload)
            telemetry_logger.debug("Webhook heartbeat acknowledged", url=self._webhook_url)
            return True
        except Exception as exc:
            telemetry_logger.error(
                "Webhook heartbeat failed",
                url=self._webhook_url,
                error=str(exc),
            )
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response


class HeartbeatReporter:
    """Asynchronous telemetry heartbeat scheduler."""

    def __init__(
        self,
        *,
        state: "WorkerState",
        cache_manager: Optional["ModelCacheManager"] = None,
        config: Optional[TelemetryConfig] = None,
        publisher: Optional[TelemetryPublisher] = None,
        network_sampler: Optional[NetworkStatsSampler] = None,
    ) -> None:
        from worker.state import WorkerState  # Local import to avoid circular

        if not isinstance(state, WorkerState):  # pragma: no cover - defensive
            raise TypeError("state must be a WorkerState instance")

        self._state = state
        self._cache_manager = cache_manager
        self._config = config or TelemetryConfig.from_env()
        self._publisher = publisher
        self._network_sampler = network_sampler or NetworkStatsSampler()
        self._task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._snapshot_lock = Lock()
        self._latest_snapshot: Optional[TelemetrySnapshot] = None

    async def start(self) -> None:
        if not self._config.enabled:
            telemetry_logger.info("Telemetry disabled; heartbeat reporter not started")
            return
        if self._task and not self._task.done():
            return
        self._shutdown = False
        self._task = asyncio.create_task(self._run())
        telemetry_logger.info(
            "Telemetry heartbeat reporter started",
            interval_seconds=self._config.interval_seconds,
        )

    async def stop(self) -> None:
        self._shutdown = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        telemetry_logger.info("Telemetry heartbeat reporter stopped")

    async def collect_snapshot(self) -> TelemetrySnapshot:
        worker_status = await self._collect_worker_status()
        cache_info = self._collect_cache_snapshot()
        gpu_info = self._collect_gpu_info()
        network_info = self._network_sampler.sample()

        snapshot = TelemetrySnapshot(
            timestamp=time.time(),
            cadence_seconds=float(self._config.interval_seconds),
            worker_id=self._config.worker_id,
            location=self._config.location,
            worker_status=worker_status,
            cache=cache_info,
            gpu=gpu_info,
            network=network_info,
        )

        with self._snapshot_lock:
            self._latest_snapshot = snapshot

        return snapshot

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._snapshot_lock:
            if not self._latest_snapshot:
                return None
            return copy.deepcopy(self._latest_snapshot.as_dict())

    async def _run(self) -> None:
        try:
            while not self._shutdown:
                try:
                    snapshot = await self.collect_snapshot()
                    payload = snapshot.as_dict()

                    if self._publisher:
                        publish_result = await self._publisher.publish(payload)
                        log_method = telemetry_logger.info if publish_result.succeeded() else telemetry_logger.warning
                        log_method(
                            "Heartbeat published",
                            targets=publish_result.as_dict(),
                            worker_id=self._config.worker_id,
                            mode=snapshot.worker_status.get("mode"),
                            cache_models=snapshot.cache.get("total_models"),
                        )
                    else:
                        telemetry_logger.debug(
                            "Heartbeat snapshot collected (no publisher configured)",
                            worker_id=self._config.worker_id,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive logging
                    telemetry_logger.error(
                        "Heartbeat cycle failed",
                        error=str(exc),
                        worker_id=self._config.worker_id,
                    )

                interval = self._config.interval_seconds
                await asyncio.sleep(interval if interval > 0 else 0)
        except asyncio.CancelledError:
            telemetry_logger.debug("Heartbeat reporter task cancelled")
            raise

    async def _collect_worker_status(self) -> Dict[str, Any]:
        from worker.state import ValidationJob  # Local import to avoid circular

        mode = (await self._state.get_mode()).value
        active_inference = await self._state.get_active_inference_count()
        pending_validation = await self._state.get_pending_validation()

        pending_payload: Optional[Dict[str, Any]] = None
        if isinstance(pending_validation, ValidationJob):
            pending_payload = {
                "assignment_id": pending_validation.assignment_id,
                "model_name_or_path": pending_validation.model_name_or_path,
                "base_model": pending_validation.base_model,
                "status": pending_validation.status.value,
            }

        return {
            "mode": mode,
            "active_inference": active_inference,
            "pending_validation": pending_payload,
        }

    def _collect_cache_snapshot(self) -> Dict[str, Any]:
        if not self._cache_manager:
            return {"total_models": 0, "active_models": 0, "models": []}

        try:
            models = self._cache_manager.list_models(include_inactive=True)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive logging
            telemetry_logger.error("Failed to list cached models", error=str(exc))
            return {"total_models": 0, "active_models": 0, "models": []}

        simplified_models = []
        for model in models:
            simplified_models.append(
                {
                    "model_id": model.get("model_id"),
                    "revision": model.get("revision"),
                    "model_type": model.get("model_type"),
                    "size_mb": round(float(model.get("size_mb", 0.0)), 3),
                    "is_active": bool(model.get("is_active", False)),
                }
            )

        return {
            "total_models": len(simplified_models),
            "active_models": sum(1 for model in simplified_models if model["is_active"]),
            "models": simplified_models,
        }

    def _collect_gpu_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"gpu_type": get_gpu_type(), "available": False}

        try:
            import torch

            available = torch.cuda.is_available()
            info["available"] = bool(available)
            if available:
                info["device_count"] = torch.cuda.device_count()
        except Exception as exc:
            info["error"] = str(exc)

        return info
