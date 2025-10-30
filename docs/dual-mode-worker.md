# Dual-mode Worker Lifecycle

This guide explains how the `DualModeWorker` orchestrates validation loops alongside the inference API service.

## Threads and events

The manager creates up to three background components:

1. **Validation loop thread** – polls FedLedger for assignments, delegates execution to `ValidationRunner`, and reports results.
2. **Inference server supervisor** – starts an ASGI server (e.g. uvicorn) and monitors it for exits so retries can be attempted.
3. **Heartbeat timer** – periodically emits telemetry payloads via `WorkerTelemetry`.

A shared `threading.Event` (`stop_event`) coordinates shutdown. All worker threads check `stop_event.is_set()` between iterations.

## Mode matrix

| Mode        | Validation loop | Inference server |
|-------------|-----------------|------------------|
| validation  | ✅               | ❌ (disabled)     |
| inference   | ❌               | ✅               |
| dual        | ✅               | ✅               |

The mode is passed to the manager on initialisation. Validation-only behaviour remains backward compatible with previous revisions where `start.sh` directly invoked `validate.py loop`.

## Inference & validation coordination

The `InferenceActivityTracker` keeps track of in-flight inference requests. When the validation loop needs exclusive access to the model or GPU (for example, when executing a large assignment) it obtains the tracker lock. Any new inference calls block until validation releases the lock to avoid resource contention.

## Telemetry integration

```python
telemetry = WorkerTelemetry(telemetry_client, worker_id="validator-01")
telemetry.emit_status(mode="dual", cache=cache.stats())
```

Each heartbeat includes:

- worker mode (validation / inference / dual)
- last assignment metadata
- cache usage summary
- GPU information when available

## Graceful shutdown

`DualModeWorker.stop()` sets the `stop_event`, waits for the validation loop to finish the current assignment, and then stops the inference server. Tests cover both fast-shutdown and long-running inference scenarios to ensure the manager never deadlocks.
