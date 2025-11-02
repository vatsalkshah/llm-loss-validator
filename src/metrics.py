import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass
class EWMA:
    alpha: float
    value: Optional[float] = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        return self.value


@dataclass
class ThroughputMeter:
    ewma: EWMA = field(default_factory=lambda: EWMA(alpha=0.3))
    last_time: Optional[float] = None
    tokens_emitted: int = 0

    def reset(self) -> None:
        self.last_time = None
        self.tokens_emitted = 0

    def tick(self, tokens: int = 1) -> float:
        now = time.perf_counter()
        if self.last_time is None:
            self.last_time = now
            self.tokens_emitted = 0
            return 0.0
        self.tokens_emitted += tokens
        elapsed = now - self.last_time
        if elapsed <= 0:
            return self.ewma.value or 0.0
        inst_tps = tokens / elapsed
        self.last_time = now
        return self.ewma.update(inst_tps)

    def snapshot(self) -> float:
        return self.ewma.value or 0.0


async def measure_network_io(interval_sec: float = 1.0) -> tuple[int, int]:
    net0 = psutil.net_io_counters()
    await asyncio.sleep(interval_sec)
    net1 = psutil.net_io_counters()
    up_bps = int((net1.bytes_sent - net0.bytes_sent) * 8 / interval_sec)
    down_bps = int((net1.bytes_recv - net0.bytes_recv) * 8 / interval_sec)
    return up_bps, down_bps


@dataclass
class HeartbeatState:
    last_tps: float = 0.0
    uplink_bps: int = 0
    downlink_bps: int = 0
    cached_models: list[str] = field(default_factory=list)


class HeartbeatService:
    def __init__(self, interval_sec: int) -> None:
        self.interval_sec = interval_sec
        self.state = HeartbeatState()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def set_cached_models(self, names: list[str]) -> None:
        self.state.cached_models = names

    def set_tps(self, tps: float) -> None:
        self.state.last_tps = tps

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                up, down = await measure_network_io(1.0)
                self.state.uplink_bps = up
                self.state.downlink_bps = down
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._stop_event.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
