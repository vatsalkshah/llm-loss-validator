# llm-loss-validator

Validator and inference worker for Hugging Face–compatible large language models used on the FLock platform. The project now supports three deployment modes:

- **Validation-only** – legacy workflow that continuously pulls assignments from FedLedger and submits loss metrics.
- **Inference-only** – runs an inference API backed by the same model management logic used by the validator.
- **Dual-mode** – launches both the validation loop and inference API side-by-side with shared caching, telemetry and heartbeat reporting.

## Table of contents

1. [Overview](#overview)
2. [Environment setup](#environment-setup)
3. [Worker modes](#worker-modes)
   - [Validation-only (legacy and still supported)](#validation-only-legacy-and-still-supported)
   - [Inference-only mode](#inference-only-mode)
   - [Dual-mode (validation--inference)](#dual-mode-validation--inference)
4. [Configuration templates](#configuration-templates)
5. [Running inside Docker](#running-inside-docker)
6. [Inference API reference](#inference-api-reference)
   - [Synchronous completions](#synchronous-completions)
   - [Streaming completions](#streaming-completions)
7. [Cache and storage management](#cache-and-storage-management)
8. [Telemetry and heartbeats](#telemetry-and-heartbeats)
9. [TLS and networking considerations](#tls-and-networking-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Optional: Installing FlashAttention](#optional-installing-flashattention)

## Overview

`src/start.sh` is the canonical launcher for every mode. It accepts command-line flags, environment variables, or an optional dotenv file. The script exposes new knobs for telemetry and TLS client authentication, cache sizing, and inference server bootstrapping.

Under the hood the validation loop still executes `python validate.py loop`, while the inference API is expected to be provided via an ASGI app (for example `src.inference.server:app`) started with `uvicorn` or a custom shell command. Both services share the Hugging Face cache and telemetry stack so that artifacts only need to be downloaded once.

## Environment setup

### Python environment

We recommend Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Key packages added for the dual-mode worker include `fastapi`, `uvicorn`, `httpx`, `aiofiles`, `psutil`, and `pyyaml`.

### Optional dependencies

- **FlashAttention** (GPU only) – see the dedicated section below.
- **CUDA/cuDNN** – required for GPU validation/inference.
- **System TLS certificates** – mount or install any private CA bundles needed for telemetry or inference TLS termination.

## Worker modes

All examples below assume the shell is located in `src/` so that `start.sh` can find `validate.py` and the sample configs.

### Validation-only (legacy and still supported)

Nothing changes for the classic validator workflow. Provide your Hugging Face token, FLock API key, and task IDs and the script will continuously request assignments from FedLedger.

```bash
cd src
HF_TOKEN=your_hf_token \
FLOCK_API_KEY=your_flock_api_key \
TASK_ID=8,9 \
./start.sh --mode validation \
  --validation_args_file validation_config.json.example \
  --auto_clean_cache true \
  --lora_only true
```

Single-run validation remains available for local smoke testing:

```bash
cd src
FLOCK_API_KEY="<your-api-key>" \
HF_TOKEN="<your-hf-token>" \
python validate.py validate \
  --model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
  --base_model qwen1.5 \
  --eval_file ./data/dummy_data.jsonl \
  --context_length 128 \
  --max_params 7000000000 \
  --local_test \
  --validation_args_file validation_config_cpu.json.example
```

All CPU/GPU examples in previous revisions remain accurate—those workflows are still supported.

### Inference-only mode

Run only the API service when you want to host model inference without validation:

```bash
cd src
HF_TOKEN=your_hf_token \
./start.sh --mode inference \
  --inference-module src.inference.server:app \
  --inference-host 0.0.0.0 \
  --inference-port 8000 \
  --telemetry-url https://telemetry.example.com/ingest \
  --telemetry-cert-path /secrets/client.crt \
  --telemetry-key-path /secrets/client.key
```

Alternatively, supply a full shell command via `--inference-command "python server.py --port 8000"` if you prefer to manage the ASGI lifecycle yourself.

### Dual-mode (validation + inference)

Dual-mode launches both the validation loop and inference API in parallel. The script supervises both processes and will exit if either one terminates unexpectedly.

```bash
cd src
./start.sh --mode dual \
  --env-file ../config-samples/dual-mode.env.example \
  --inference-module src.inference.server:app \
  --cache-dir /data/hf-cache \
  --cache-limit-gb 60
```

Any validation-only tooling continues to work. Dual-mode simply adds the API server and telemetry while keeping the validation loop unchanged.

## Configuration templates

Sample configuration files live under `config-samples/`:

| File | Description |
| --- | --- |
| `config-samples/validation.env.example` | Minimal `.env` for the legacy validation loop. |
| `config-samples/inference.env.example` | `.env` tailored for inference-only deployments. |
| `config-samples/dual-mode.env.example` | `.env` for the dual-mode worker (validation + inference). |
| `config-samples/dual-mode.yaml.example` | YAML blueprint for orchestration systems (convert to env vars before using `start.sh`). |
| `src/.env.example` | Comprehensive baseline illustrating every knob, useful for bespoke setups. |

Copy the appropriate template, adjust the placeholders, then point `start.sh` at the file with `--env-file` or `source` it before launching the worker.

## Running inside Docker

### CPU image

```bash
docker build -t flock-validator:cpu .
docker run --rm \
  -e HF_TOKEN=your_hf_token \
  -e FLOCK_API_KEY=your_flock_api_key \
  -e TASK_ID=8,9 \
  -e WORKER_MODE=dual \
  -e INFERENCE_MODULE=src.inference.server:app \
  -p 8000:8000 \
  flock-validator:cpu \
  --mode dual --validation_args_file validation_config_cpu.json.example
```

### GPU image

```bash
docker build -f Dockerfile-gpu -t flock-validator:gpu .
docker run --rm --gpus all \
  -e HF_TOKEN=your_hf_token \
  -e FLOCK_API_KEY=your_flock_api_key \
  -e TASK_ID=8,9 \
  -e WORKER_MODE=validation \
  flock-validator:gpu \
  --mode validation --validation_args_file validation_config.json.example
```

Expose port 8000 (or whatever you configure) when running inference. The Dockerfiles now export `WORKER_MODE`, `IS_DOCKER_CONTAINER`, and default cache limits—you can override these at runtime.

## Inference API reference

The default ASGI app exposes OpenAI-style endpoints.

### Synchronous completions

`POST /v1/generate`

```bash
curl -s http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Summarise the following text...",
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.95,
        "stop": ["</s>"]
      }' | jq
```

Example response:

```json
{
  "request_id": "2c0f8682-3bd8-4a32-8f73-12c4a0f0f31b",
  "output": "Here is your summary...",
  "finish_reason": "stop",
  "latency_ms": 842,
  "tokens_generated": 215
}
```

### Streaming completions

`POST /v1/generate/stream` emits Server-Sent Events (SSE):

```bash
curl -N http://localhost:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "List three prime numbers.", "max_new_tokens": 32}'
```

The stream yields chunks such as:

```
data: {"delta": "2", "token": 1234}

data: {"delta": ", ", "token": 56}
...

data: {"event": "end", "finish_reason": "stop"}

```

Clients should close the connection after receiving the terminal `event=end` message. If telemetry is enabled, request/latency metrics are pushed to the configured endpoint after every completion.

## Cache and storage management

- `CACHE_DIR` / `HF_HOME` – set the shared Hugging Face cache directory. Mount persistent storage here for faster warm starts.
- `CACHE_LIMIT_BYTES` / `CACHE_LIMIT_GB` – hard cap the cache size. The launcher converts GiB to bytes if only `CACHE_LIMIT_GB` is supplied.
- `AUTO_CLEAN_CACHE` – when `true`, removes stale models between assignments while keeping base models from `SUPPORTED_BASE_MODELS`.
- `LORA_ONLY` – set to `false` if you want to validate full fine-tuned models.
- `HF_TOKEN` – required for gated repos. The script exits early if the token is missing.

## Telemetry and heartbeats

The worker interacts with FLock’s FedLedger service in two ways:

1. **Assignment lifecycle** – the validation loop requests assignments and submits loss metrics through `client/fed_ledger.py`.
2. **Heartbeats** – optional periodic status reports. Configure with:
   - `HEARTBEAT_ENDPOINT` – override the default FedLedger heartbeat route.
   - `HEARTBEAT_INTERVAL` – seconds between heartbeat payloads.

Enable telemetry export by supplying `TELEMETRY_URL`. Optional TLS knobs (`TELEMETRY_TLS_CERT_PATH`, `TELEMETRY_TLS_KEY_PATH`, `TELEMETRY_TLS_CA_PATH`) support mTLS or private CAs. When telemetry is disabled the launcher explicitly sets `TELEMETRY_ENABLED=false` so downstream code can short-circuit.

## TLS and networking considerations

You have several deployment patterns:

1. **External termination** – place an ingress proxy (nginx, Traefik, Load Balancer) in front of the container. Leave `TLS_ENABLED=false` and configure TLS at the proxy layer.
2. **Direct termination** – mount certificates into the container and set `TLS_ENABLED=true` with `TLS_CERT_PATH` and `TLS_KEY_PATH`. Combine with `--inference-extra "--ssl-keyfile ... --ssl-certfile ..."` when launching `uvicorn`.
3. **mTLS telemetry** – regardless of inference TLS approach, you can configure telemetry to use client certs (see env vars above) while the API remains behind a proxy.

Always expose only the inference port(s) needed and restrict access to the validation loop with firewall rules—it only talks to FedLedger.

## Troubleshooting

| Symptom | Suggested checks |
| --- | --- |
| Validation loop exits immediately | Confirm `TASK_ID` points to active tasks and the machine clock is reasonably accurate. Inspect logs for FedLedger HTTP status codes. |
| Inference API returns 409/429 | The rate limiter may be tripping—adjust `MAX_CONCURRENT_REQUESTS` or ensure batching logic is applied inside the inference server. |
| Excessive cache churn | Increase `CACHE_LIMIT_GB` or disable `AUTO_CLEAN_CACHE`. Validate that the cache directory is writable and has enough free space. |
| Telemetry failures | Verify TLS materials, check `TELEMETRY_URL`, and ensure outbound network connectivity. The launcher logs configuration before startup to help spot issues. |
| TLS handshake errors | When terminating TLS inside the worker, confirm the certificate chain and key permissions. For mTLS, ensure both cert and key are provided. |
| GPU not detected | Set `CUDA_VISIBLE_DEVICES` and double-check that the NVIDIA drivers/containers expose the device to the worker. |

## Optional: Installing FlashAttention

FlashAttention is a fast and memory-efficient attention mechanism that can be beneficial for large models. However, it can be tricky to compile depending on your GPU setup.

### Advantages
- **Memory Efficiency** – reduces memory usage significantly, allowing for longer sequence lengths.
- **Speed** – provides a speedup over standard attention mechanisms, especially on GPUs with high memory bandwidth.

### Installation guide
1. Ensure CUDA Toolkit ≥ 11.7 is installed.
2. Install FlashAttention:
   ```bash
   pip install flash-attn --no-build-isolation
   ```
   Alternatively, build from source:
   ```bash
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention
   python setup.py install
   ```

### Using FlashAttention 2

For models like Qwen that automatically support FlashAttention, just installing the binary suffices. For other models that support FlashAttention 2, enable it by adding `attn_implementation="flash_attention_2"` in the `AutoModelForCausalLM.from_pretrained()` call within [`validate.py`](src/validate.py).

For more information refer to the [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention).
