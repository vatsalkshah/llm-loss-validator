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
   - Periodic status reporting
   - Logs worker health metrics
   - Configurable reporting interval

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
- `--lora_only`: Only validate LoRA models (default: `True`)

### Environment Variables

Required:
- `FLOCK_API_KEY`: Flock API key for FedLedger
- `HF_TOKEN`: HuggingFace authentication token

Optional:
- `CACHE_DIR`: Model cache directory
- `CACHE_MAX_SIZE_GB`: Maximum cache size in GB
- `CACHE_EVICTION_STRATEGY`: Eviction strategy (LRU, FIFO, SIZE)

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

The telemetry reporter logs worker status at regular intervals:

```
Worker Status:
  Mode: inference_active
  Active Inference Requests: 2
  Pending Validation: None
```

During validation:

```
Worker Status:
  Mode: validation_active
  Active Inference Requests: 0
  Pending Validation: assignment_12345 (status: running)
```

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
