import asyncio

import pytest

from core.telemetry import (
    HeartbeatReporter,
    PublishResult,
    TelemetryConfig,
)
from worker.state import ValidationJob, WorkerMode, WorkerState


class StubCacheManager:
    def list_models(self, include_inactive: bool = True):  # pragma: no cover - simple stub
        return [
            {
                "model_id": "stub/model",
                "revision": "main",
                "model_type": "base",
                "size_mb": 128.0,
                "is_active": True,
            }
        ]


class FixedSampler:
    def sample(self):
        return {"download_mbps": 1.0, "upload_mbps": 2.5}


class RecordingPublisher:
    def __init__(self, succeed: bool = True):
        self.calls = 0
        self.payloads = []
        self._succeed = succeed

    async def publish(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        return PublishResult(fed_ledger=self._succeed, webhook=None)


@pytest.mark.asyncio
async def test_collect_snapshot_includes_expected_sections():
    state = WorkerState()
    started = await state.start_inference_request()
    assert started is True
    await state.queue_validation(
        ValidationJob(
            assignment_id="assignment-123",
            model_name_or_path="stub/model",
            base_model="stub",
            eval_file="/tmp/eval.jsonl",
            context_length=128,
            max_params=1_000_000,
        )
    )

    reporter = HeartbeatReporter(
        state=state,
        cache_manager=StubCacheManager(),
        config=TelemetryConfig(
            enabled=True,
            interval_seconds=5.0,
            webhook_url=None,
            location="PDX",
            worker_id="worker-42",
        ),
        publisher=None,
        network_sampler=FixedSampler(),
    )

    snapshot = await reporter.collect_snapshot()
    payload = snapshot.as_dict()

    assert payload["cadence_seconds"] == 5.0
    assert payload["status"]["mode"] == WorkerMode.VALIDATION_PENDING.value
    assert payload["status"]["active_inference"] == 1
    assert payload["status"]["pending_validation"]["assignment_id"] == "assignment-123"
    assert payload["cache"]["total_models"] == 1
    assert payload["network"] == {"download_mbps": 1.0, "upload_mbps": 2.5}
    assert payload["worker"]["id"] == "worker-42"
    assert payload["worker"]["location"] == "PDX"
    assert "available" in payload["gpu"]


@pytest.mark.asyncio
async def test_heartbeat_reporter_schedules_publisher_calls():
    state = WorkerState()
    publisher = RecordingPublisher()

    reporter = HeartbeatReporter(
        state=state,
        cache_manager=StubCacheManager(),
        config=TelemetryConfig(
            enabled=True,
            interval_seconds=0.1,
            webhook_url=None,
            location=None,
            worker_id="stub-worker",
        ),
        publisher=publisher,
        network_sampler=FixedSampler(),
    )

    await reporter.start()
    await asyncio.sleep(0.3)
    await reporter.stop()

    assert publisher.calls >= 2
    latest = reporter.get_latest_snapshot()
    assert latest is not None
    assert latest["status"]["mode"] == WorkerMode.IDLE.value
