import asyncio

import pytest

from worker.inference_guard import InferenceActivityTracker, InferenceBusyError
from worker.state import WorkerState, ValidationJob, ValidationJobStatus, WorkerMode


@pytest.mark.asyncio
async def test_tracker_context_manager_tracks_active_requests():
    state = WorkerState()
    tracker = InferenceActivityTracker(state)

    assert await state.get_active_inference_count() == 0
    assert await state.get_mode() == WorkerMode.IDLE

    async with tracker.track():
        assert await state.get_active_inference_count() == 1
        assert await state.get_mode() == WorkerMode.INFERENCE_ACTIVE

    assert await state.get_active_inference_count() == 0
    assert await state.get_mode() == WorkerMode.IDLE


@pytest.mark.asyncio
async def test_validation_preempts_new_inference_requests():
    state = WorkerState()
    tracker = InferenceActivityTracker(state)

    # Start an inference request
    await tracker.acquire()

    # Queue validation job
    job = ValidationJob(
        assignment_id="a1",
        model_name_or_path="model",
        base_model="base",
        eval_file="/tmp/eval.json",
        context_length=1024,
        max_params=10,
    )
    await state.queue_validation(job)

    # New inference requests should be blocked
    with pytest.raises(InferenceBusyError):
        await tracker.acquire()

    # Release the first request
    await tracker.release()

    # Start validation and mark complete
    await state.start_validation()
    await state.complete_validation(success=True)

    assert await state.get_mode() == WorkerMode.IDLE
    assert await state.get_pending_validation() is None


@pytest.mark.asyncio
async def test_validation_waits_for_inflight_inference_to_clear():
    state = WorkerState()
    tracker = InferenceActivityTracker(state)

    await tracker.acquire()

    job = ValidationJob(
        assignment_id="a2",
        model_name_or_path="model",
        base_model="base",
        eval_file="/tmp/eval.json",
        context_length=1024,
        max_params=10,
    )
    await state.queue_validation(job)

    waiter = asyncio.create_task(state.wait_for_inference_clear())
    await asyncio.sleep(0)
    assert not waiter.done()

    await tracker.release()
    await asyncio.sleep(0)
    assert waiter.done()

    await state.start_validation()
    await state.complete_validation(success=True)

    assert job.status == ValidationJobStatus.COMPLETED
    assert await state.get_mode() == WorkerMode.IDLE
