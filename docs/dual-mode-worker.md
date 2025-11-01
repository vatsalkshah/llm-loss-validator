# Dual-Mode Worker Documentation

## Overview

The dual-mode worker orchestrator coordinates validation assignments and inference serving within a single process using asyncio. It ensures validation jobs get priority while allowing in-flight inference requests to complete before pre-emption.

## Architecture

The dual-mode worker consists of several components:

### Components

1. **WorkerState** (`worker/state.py`)
   - Thread-safe state manager
   - Tracks active inference requests
   - Manages validation job queue
   - Coordinates priority scheduling

2. **InferenceActivityTracker** (`worker/inference_guard.py`)
   - API for inference server to signal active requests
   - Provides context manager for request lifecycle
   - Blocks new requests when validation is pending

3. **ValidationPoller** (`worker/validation_poller.py`)
   - Polls FedLedger every 3 minutes for assignments
   - Hydrates models via ModelCacheManager
   - Dispatches to ValidationRunner
   - Coordinates with WorkerState for priority

4. **TelemetryReporter** (`worker/telemetry.py`)
   - Wraps `core.telemetry.HeartbeatReporter` for background heartbeats
   - Publishes structured telemetry to FedLedger and optional HTTPS webhooks
   - Exposes the latest snapshot for other components and logs outcomes

5. **DualModeWorker** (`worker/manager.py`)
   - Main orchestrator class
   - Boots inference server, poller, telemetry
   - Manages clean startup/shutdown
   - Signal handling for graceful termination

### Priority Scheduling

The worker implements validation priority through the following mechanism:

1. **Normal Operation**: Inference requests proceed normally
2. **Validation Queued**: New inference requests are blocked
3. **In-Flight Completion**: Active inference requests complete
4. **Validation Execution**: Validation runs with exclusive access
5. **Resume Inference**: New inference requests unblocked

## Usage

### Quick Start with `start.sh`

The helper script `src/start.sh` wraps the Click CLI and wiring for telemetry, cache controls, and TLS. From the repository root:

```bash
cd src
./start.sh --mode dual \
  --hf_token "$HF_TOKEN" \
  --flock_api_key "$FLOCK_API_KEY" \
  --task_id 12,27
```

For repeatable deployments, copy [`configs/dual-mode.env.example`](../configs/dual-mode.env.example) to a managed location and `source` it (or pass with `docker --env-file`), then invoke `./start.sh --mode "$START_MODE"`.

### Starting the Dual-Mode Worker

```bash
# Basic usage
python src/validate.py worker --task_id 1,2,3

# With custom configuration
python src/validate.py worker \
  --task_id 1,2,3 \
  --validation_args_file validation_config.json \
  --inference_host 0.0.0.0 \
  --inference_port 8000 \
  --polling_interval 180 \
  --telemetry_interval 60 \
  --lora_only True
```

### Options

- `--task_id`: Comma-separated task IDs to validate (required)
- `--validation_args_file`: Path to validation config JSON (default: `validation_config.json.example`)
- `--inference_host`: Host for inference server (default: `0.0.0.0`)
- `--inference_port`: Port for inference server (default: `8000`)
- `--polling_interval`: Validation polling interval in seconds (default: `180`)
- `--telemetry_interval`: Telemetry reporting interval in seconds (default: `60`)
- `--telemetry_webhook`: HTTPS endpoint for heartbeat telemetry (optional)
- `--telemetry_location`: Location metadata to attach to heartbeat payloads (optional)
- `--telemetry_worker_id`: Override worker identifier in telemetry payloads (optional)
- `--lora_only`: Only validate LoRA models (default: `True`)

### Environment Variables

Required:
- `FLOCK_API_KEY`: Flock API key for FedLedger
- `HF_TOKEN`: HuggingFace authentication token

Optional:
- `CACHE_DIR`, `CACHE_MAX_SIZE_GB`, `CACHE_EVICTION_STRATEGY`, `CACHE_AUTO_EVICT`: Cache tuning knobs shared with the inference server.
- `VALIDATION_ARGS_FILE`: Default validation config JSON consumed by the worker.
- `START_MODE`: Default mode leveraged by `start.sh` (defaults to `dual`).
- `INFERENCE_HOST`, `INFERENCE_PORT`, `POLLING_INTERVAL`: Command defaults when launching via `start.sh`.
- `REQUIRE_API_KEY`, `API_KEY`, `API_KEY_HEADER`: Enable API key enforcement for inference traffic.
- `TLS_ENABLED`, `TLS_CERT_PATH`, `TLS_KEY_PATH`: Enable TLS for the embedded inference server.
- `REJECT_NON_CACHED_MODELS`: Reject inference requests when the model is absent from the cache.
- `TELEMETRY_ENABLED`, `TELEMETRY_INTERVAL_SECONDS`, `TELEMETRY_WEBHOOK_URL`, `TELEMETRY_LOCATION`, `TELEMETRY_WORKER_ID`: Telemetry configuration surface.

## Backward Compatibility

The existing `loop` command continues to work as before:

```bash
# Original validation loop (still supported)
python src/validate.py loop --task_id 1
```

The `worker` command is a superset that adds inference serving capability.

## Inference API

When the dual-mode worker is running, an OpenAI-compatible inference API is available:

```bash
# List available models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

During validation, new inference requests will receive a `503 Service Unavailable` response with the message "Inference service temporarily unavailable (validation in progress)".

## Signal Handling

The worker handles the following signals for graceful shutdown:

- `SIGINT` (Ctrl+C): Initiate graceful shutdown
- `SIGTERM`: Initiate graceful shutdown

Shutdown sequence:
1. Stop accepting new validation assignments
2. Wait for current validation to complete
3. Stop validation poller
4. Stop telemetry reporter
5. Stop inference server
6. Clean up resources

## Integration Tests

Run integration tests to validate priority scheduling behavior:

```bash
pytest tests/worker/test_dual_mode_worker.py -v
```

## Monitoring

Telemetry payloads are emitted at the configured cadence and logged with structured metadata:

```
2024-05-01 12:00:00.123 | component=telemetry | INFO  | Heartbeat published | targets={'fed_ledger': True, 'webhook': None} mode=idle cache_models=2
```

The underlying `core.telemetry.HeartbeatReporter` captures worker mode, pending validation, cache inventory, GPU availability, and estimated network throughput. Other components can call `TelemetryReporter.get_latest_snapshot()` to inspect the most recent payload without waiting for the next publish cycle.

## Operational Considerations

### TLS & Network Topology
- When running inside a single container, enable TLS by exporting `TLS_ENABLED=true` and pointing `TLS_CERT_PATH`/`TLS_KEY_PATH` to mounted files.
- If a reverse proxy terminates TLS, keep `TLS_ENABLED=false` and forward plain HTTP traffic over the loopback network.
- Disable proxy buffering for `/v1/chat/completions` when streaming responses to avoid SSE truncation.

### FedLedger Expectations
- Assignment polling obeys FedLedger rate limits; adjust `POLLING_INTERVAL` (or CLI flag) if you consistently see rate limit warnings.
- Successful validations invoke `FedLedger.submit_validation_result`; failures should be marked explicitly so the coordinator can reassign them.
- Heartbeat payloads are acknowledged by FedLedger—monitor the `targets` log field for `fed_ledger=true` to confirm receipt.

### Telemetry & Cache Health
- Webhook fan-out requires HTTPS; failures surface as warnings with retry detail. Validate certificates when testing staging endpoints.
- Cache utilisation is emitted in heartbeat payloads (`cache.utilization_percent`). Sudden drops typically indicate eviction triggered by policy.
- Combine telemetry snapshots with `ModelCacheManager.get_cache_stats()` for deeper diagnostics of eviction behaviour.

## Troubleshooting

### Worker fails to start

- Check that `FLOCK_API_KEY` and `HF_TOKEN` are set
- Verify validation config file exists
- Check port availability for inference server

### Inference requests hang

- Check if validation is in progress (503 response expected)
- Verify worker mode via telemetry logs
- Check for errors in validation execution

### Validation jobs not executing

- Verify task IDs are correct
- Check FedLedger connectivity
- Review polling interval and rate limits
- Check for validation errors in logs

## Migration from Loop Command

To migrate from the `loop` command to `worker`:

1. Identify your current `loop` command arguments
2. Add inference server configuration
3. Update your deployment scripts

Before:
```bash
python src/validate.py loop --task_id 1,2,3 --validation_args_file config.json
```

After:
```bash
python src/validate.py worker --task_id 1,2,3 --validation_args_file config.json
```

The `worker` command maintains the same validation behavior while adding inference capability.
