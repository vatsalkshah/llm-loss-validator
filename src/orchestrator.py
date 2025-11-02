from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from loguru import logger

from src.config import get_config
from src.inference import run_generate, stream_generate
from src.metrics import HeartbeatService, ThroughputMeter
from src.core.gpu_utils import get_gpu_type
from src.core.model_registry import get_registry
from src.client.fed_ledger import FedLedger


class Orchestrator:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.gpu_lock = asyncio.Lock()
        self.validation_pending = asyncio.Event()
        self.validation_running = asyncio.Event()
        self.heartbeat = HeartbeatService(interval_sec=cfg.heartbeat_interval_sec)
        self._watcher_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        # Start heartbeat updater
        self.heartbeat.start()
        # Start validation watcher if configured
        if self.cfg.validation_task_id and self.cfg.flock_api_key:
            self._watcher_task = asyncio.create_task(self._validation_watcher())
        else:
            logger.info("Validation watcher disabled (VALIDATION_TASK_ID or FLOCK_API_KEY missing)")

    async def stop(self) -> None:
        await self.heartbeat.stop()
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass

    async def stream_inference(self, model_id: str, prompt: str, gen: Dict[str, Any]) -> AsyncGenerator[str, None]:
        # Do not start new inference if validation is pending or running
        while self.validation_pending.is_set() or self.validation_running.is_set():
            await asyncio.sleep(0.1)

        async with self.gpu_lock:
            # Keep cached models list in heartbeat
            registry = get_registry()
            self.heartbeat.set_cached_models(registry.list_loaded())
            meter = ThroughputMeter()
            async for token in stream_generate(model_id, prompt, gen):
                meter.tick(1)
                self.heartbeat.set_tps(meter.snapshot())
                yield token

    async def run_inference(self, model_id: str, prompt: str, gen: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        while self.validation_pending.is_set() or self.validation_running.is_set():
            await asyncio.sleep(0.1)
        async with self.gpu_lock:
            registry = get_registry()
            self.heartbeat.set_cached_models(registry.list_loaded())
            return await asyncio.to_thread(run_generate, model_id, prompt, gen)

    async def _validation_watcher(self) -> None:
        # Import validate module with compatibility for its internal absolute imports
        import os
        import sys
        sys.path.append(os.path.dirname(__file__))
        import validate as validate_module  # type: ignore
        ASSIGNMENT_LOOKUP_INTERVAL = validate_module.ASSIGNMENT_LOOKUP_INTERVAL
        fed_ledger = FedLedger(self.cfg.flock_api_key)
        task_id = self.cfg.validation_task_id
        assert task_id is not None
        logger.info("Validation watcher started")
        while True:
            try:
                resp = await asyncio.to_thread(fed_ledger.request_validation_assignment, task_id)
                if resp is not None and resp.status_code == 200:
                    payload = resp.json()
                    self.validation_pending.set()
                    async with self.gpu_lock:
                        self.validation_running.set()
                        self.validation_pending.clear()
                        logger.info("Running validation assignment")
                        await asyncio.to_thread(
                            validate_module.run_validation_assignment,
                            payload,
                            "validation_config.json.example",
                            True,
                        )
                        self.validation_running.clear()
                await asyncio.sleep(ASSIGNMENT_LOOKUP_INTERVAL)
            except Exception as e:
                logger.exception(f"Validation watcher error: {e}")
                await asyncio.sleep(ASSIGNMENT_LOOKUP_INTERVAL)

    def heartbeat_snapshot(self) -> Dict[str, Any]:
        gpu_type = get_gpu_type()
        registry = get_registry()
        return {
            "nodes": 1,
            "gpu_type": gpu_type,
            "last_tps": self.heartbeat.state.last_tps,
            "uplink_bps": self.heartbeat.state.uplink_bps,
            "downlink_bps": self.heartbeat.state.downlink_bps,
            "location": self.cfg.node_location,
            "cached_models": registry.list_loaded(),
        }


# Singleton orchestrator
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
