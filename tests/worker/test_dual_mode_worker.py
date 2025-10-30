import threading
import time

from src.worker.manager import DualModeWorker
from src.worker.state import WorkerMode


class DummyRunner:
    def __init__(self):
        self.calls = []

    def run(self, payload):
        self.calls.append(payload.get("id"))


class DummyLedger:
    def __init__(self):
        self.responses = []

    def request_validation_assignment(self, task_id):
        data = {"id": f"assignment-{len(self.responses)}", "task_id": task_id}
        self.responses.append(data)
        return data


def test_dual_mode_worker_runs_validation():
    ledger = DummyLedger()
    runner = DummyRunner()

    worker = DualModeWorker(
        mode=WorkerMode.VALIDATION,
        validation_runner=runner,
        validation_task_id="123",
        fed_ledger=ledger,
    )

    worker.start()
    time.sleep(0.1)
    worker.stop()

    assert runner.calls, "validation runner should be invoked"


def test_dual_mode_worker_runs_inference_callable():
    events = []

    def inference_server():
        events.append("started")

    worker = DualModeWorker(
        mode=WorkerMode.INFERENCE,
        validation_runner=lambda payload: payload,
        inference_server=inference_server,
    )
    worker.start()
    time.sleep(0.05)
    worker.stop()

    assert events == ["started"]
