# llm-loss-validator

Validator that computes validation loss for Hugging Face-compatible LLMs and can optionally serve inference traffic via an OpenAI-compatible API.

## Features

- **Validation Service**: Compute validation losses for Hugging Face-compatible LLMs and submit results back to FedLedger.
- **Dual-Mode Worker**: Coordinate validation assignments and inference serving with priority scheduling and telemetry.
- **Inference API**: OpenAI-compatible REST API with synchronous and streaming responses, API key protection, and optional TLS.
- **Model Cache Management**: Configurable cache directory, size limits, and eviction strategies shared across validation and inference.
- **Heartbeat Telemetry**: Structured telemetry reporting with FedLedger fan-out and optional HTTPS webhooks.
- **LoRA Support**: Automatic base model detection and adapter merging for LoRA submissions.

## Environment Setup

We recommend using `conda` (or any virtual environment) to manage Python dependencies:

```bash
conda create -n llm-loss-validator python==3.10.12
conda activate llm-loss-validator
pip install -r requirements.txt
```

If you prefer `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (`start.sh`)

The orchestration script lives at `src/start.sh` and now supports validation-only, inference-only, and dual-mode execution.

```bash
# From the repository root
cd src
./start.sh \
  --mode dual \
  --hf_token "<your_hf_token>" \
  --flock_api_key "<your_flock_api_key>" \
  --task_id 8,9 \
  --inference_port 9000 \
  --telemetry_webhook https://telemetry.example.com/heartbeat
```

Key flags:

- `--mode dual|validation|inference`: Choose operation mode (defaults to `dual` or `START_MODE` env var).
- `--hf_token`, `--flock_api_key`, `--task_id`: Supply credentials/assignment IDs or set the corresponding environment variables.
- `--inference_host`, `--inference_port`, `--polling_interval`, `--telemetry_*`: Fine-tune dual-mode behaviour.
- `--server_host`, `--server_port`, `--server_workers`, `--reload`: Configure inference-only runs.
- Cache, TLS, and API key settings can be supplied as CLI flags or environment variables.

Sample `.env` templates are provided in `configs/`—see [Configuration Templates](#configuration-templates).

## Modes of Operation

### Dual-Mode Worker (Validation + Inference)

Dual-mode operation keeps validation as the priority workload while serving inference requests in the background.

```bash
# Start via helper script (preferred)
cd src
./start.sh --mode dual \
  --hf_token "$HF_TOKEN" \
  --flock_api_key "$FLOCK_API_KEY" \
  --task_id 12,27 \
  --inference_host 0.0.0.0 \
  --inference_port 8000 \
  --telemetry_interval 60
```

Or invoke the Click CLI directly:

```bash
cd src
python validate.py worker \
  --task_id 12,27 \
  --validation_args_file validation_config.json.example \
  --inference_host 0.0.0.0 \
  --inference_port 8000 \
  --polling_interval 180 \
  --telemetry_interval 60
```

Key options:

- `--telemetry_webhook`, `--telemetry_location`, `--telemetry_worker_id`: Augment heartbeat payloads.
- `--lora_only`: Restrict validation to LoRA submissions when desired.
- TLS settings (`TLS_ENABLED`, `TLS_CERT_PATH`, `TLS_KEY_PATH`) secure the embedded inference server.
- API key enforcement can be enabled with `REQUIRE_API_KEY=true` and `API_KEY=<value>`.

### Validation-Only (Loop Command Still Supported)

The legacy `loop` workflow remains fully supported. Use `--mode validation` to keep prior behaviour while benefiting from the refreshed configuration knobs.

```bash
cd src
./start.sh --mode validation \
  --hf_token "$HF_TOKEN" \
  --flock_api_key "$FLOCK_API_KEY" \
  --task_id 8,9 \
  --auto_clean_cache false \
  --lora_only true
```

Manual one-off validation runs are also available:

```bash
# CPU local test
cd src
FLOCK_API_KEY="$FLOCK_API_KEY" python validate.py validate \
  --model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
  --base_model qwen1.5 \
  --eval_file ./data/dummy_data.jsonl \
  --context_length 128 \
  --max_params 7000000000 \
  --local_test \
  --validation_args_file validation_config_cpu.json.example

# GPU run submitting to FedLedger
cd src
CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="$FLOCK_API_KEY" python validate.py validate \
  --model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
  --base_model qwen1.5 \
  --eval_file ./data/dummy_data.jsonl \
  --context_length 128 \
  --max_params 7000000000 \
  --assignment_id <assignment-id> \
  --validation_args_file validation_config.json.example
```

Use `--local_test` to verify the pipeline without contacting FedLedger.

### Inference-Only Server

Run the OpenAI-compatible API without participating in validation:

```bash
cd src
./start.sh --mode inference --hf_token "$HF_TOKEN" --server_port 9000
```

Or invoke the convenience runner:

```bash
cd src
python server/run_server.py --host 0.0.0.0 --port 9000 --workers 2
```

Set TLS and API key configuration via environment variables (`TLS_ENABLED`, `TLS_CERT_PATH`, `TLS_KEY_PATH`, `REQUIRE_API_KEY`, `API_KEY`, `API_KEY_HEADER`).

## Inference API Usage Examples

**Non-streaming**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 150
  }'
```

**Streaming (Server-Sent Events)**

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true,
    "max_tokens": 200
  }'
```

For client SDK examples, refer to `docs/inference-api.md`.

## Cache Manager Configuration

| Variable | Description |
| --- | --- |
| `CACHE_DIR` | Directory for Hugging Face model cache (default `~/.cache/huggingface`). |
| `CACHE_MAX_SIZE_GB` | Maximum on-disk cache size in gigabytes. |
| `CACHE_EVICTION_STRATEGY` | Eviction policy: `LRU`, `FIFO`, or `SIZE`. |
| `CACHE_AUTO_EVICT` | Automatically evict models when cache exceeds the limit (`true`/`false`). |
| `CACHE_ENABLED` | Toggle cache manager on/off (`true` by default). |
| `REJECT_NON_CACHED_MODELS` | Reject inference requests for models that are not cached locally. |

Detailed guidance lives in [`docs/cache_manager_integration.md`](docs/cache_manager_integration.md).

## Telemetry & Heartbeat Reporting

Telemetry is controlled through CLI flags or environment variables:

- `TELEMETRY_ENABLED`: Disable heartbeats entirely by setting to `false`.
- `TELEMETRY_INTERVAL_SECONDS`: Override cadence (seconds) between reports.
- `TELEMETRY_WEBHOOK_URL`: HTTPS destination in addition to FedLedger.
- `TELEMETRY_LOCATION`, `TELEMETRY_WORKER_ID`: Add location/identifier metadata.

Heartbeat payloads include worker mode, cache inventory, GPU availability, and estimated network throughput. Expect to see success/failure logs every reporting interval.

## TLS & Networking Considerations

- Enable built-in TLS by setting `TLS_ENABLED=true` along with `TLS_CERT_PATH` and `TLS_KEY_PATH`.
- When offloading TLS to a proxy (nginx, Caddy, ALB), keep `TLS_ENABLED=false` and terminate HTTPS at the edge.
- Ensure certificates are mounted read-only and refreshed atomically when rotated.
- For streaming workloads behind reverse proxies, disable buffering to avoid SSE truncation.

## FedLedger Interactions

1. **Assignment polling**: The worker (or `loop`) command polls FedLedger for validation assignments using the configured `TASK_ID` list.
2. **Result submission**: Completed validations submit losses (or mark failures) via the FedLedger client.
3. **Heartbeats**: Telemetry reports worker health, cache state, and mode; webhook delivery status is logged to aid monitoring.
4. **Cache coordination**: Optional cache state reports and commands are exposed through the cache manager integration helpers.

## Troubleshooting

- **Cache pressure**: Review `CACHE_MAX_SIZE_GB`, eviction strategy, and `CACHE_AUTO_EVICT`. Use `cache_manager.get_cache_stats()` for diagnostics.
- **Telemetry failures**: Verify `TELEMETRY_WEBHOOK_URL` uses HTTPS and returns 2xx responses. Heartbeat logs include per-target success flags.
- **FedLedger rate limits**: The loop mode backs off automatically—monitor logs for retries and adjust `POLLING_INTERVAL` if necessary.
- **TLS errors**: Ensure certificate/key paths are accessible inside the container and match the configured filenames.

More operational details are captured in [`docs/dual-mode-worker.md`](docs/dual-mode-worker.md) and [`docs/inference-api.md`](docs/inference-api.md).

## Configuration Templates

Starter environment files live under `configs/`:

- `configs/validation.env.example`: Validation-only loop worker.
- `configs/dual-mode.env.example`: Combined validation + inference worker.
- `configs/inference.env.example`: Standalone inference API server.

Copy the template that matches your deployment and adjust secrets/paths:

```bash
cp configs/dual-mode.env.example .env
# edit values as needed
source .env
cd src
./start.sh --mode "$START_MODE"
```

For containers, pass the file with `docker run --env-file configs/dual-mode.env.example ...`.

## Additional Documentation

- [`docs/dual-mode-worker.md`](docs/dual-mode-worker.md): Scheduler architecture, telemetry lifecycle, and migration guidance.
- [`docs/inference-api.md`](docs/inference-api.md): API reference, authentication, streaming examples, and TLS strategies.
- [`docs/cache_manager_integration.md`](docs/cache_manager_integration.md): Cache tuning, eviction workflows, and FedLedger integration examples.
- [`docs/VALIDATION_ARCHITECTURE.md`](docs/VALIDATION_ARCHITECTURE.md): End-to-end validation pipeline details.

## Optional: Installing FlashAttention

FlashAttention can improve throughput and memory efficiency for large models.

### Advantages
- **Memory Efficiency**: Reduced activation memory enables longer sequence lengths.
- **Speed**: Noticeable speedups on GPUs with high memory bandwidth.

### Installation Guide

1. Ensure CUDA 11.7+ is installed.
2. Install from PyPI:

   ```bash
   pip install flash-attn --no-build-isolation
   ```

   Or build from source:

   ```bash
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention
   python setup.py install
   ```

### Using FlashAttention 2

For models such as Qwen that natively support FlashAttention 2, setting `attn_implementation="flash_attention_2"` when instantiating `AutoModelForCausalLM` is sufficient. See [`src/validate.py`](src/validate.py) for integration points.

For more information, visit the [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention).
