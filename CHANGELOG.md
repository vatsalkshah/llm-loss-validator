# Changelog

All notable changes to llm-loss-validator are documented in this file.

## [Unreleased] - Dual-mode worker support

### Added

#### Worker modes
- **Inference-only mode** – run the worker as a pure inference API server without validation.
- **Dual-mode** – run both validation loop and inference API simultaneously with shared caching.
- **Validation-only mode** remains the default (unchanged legacy behavior).

#### Configuration
- `start.sh` now accepts `--mode <validation|inference|dual>` flag.
- `--env-file` option to load configuration from dotenv files.
- `--inference-module` and `--inference-command` for flexible API server bootstrapping.
- `--inference-host`, `--inference-port`, and `--inference-extra` for ASGI server configuration.
- Cache management flags: `--cache-dir`, `--cache-limit-bytes`, `--cache-limit-gb`.
- Telemetry flags: `--telemetry-url`, `--telemetry-cert-path`, `--telemetry-key-path`, `--telemetry-ca-path`.
- Heartbeat configuration: `--heartbeat-endpoint`, `--heartbeat-interval`.

#### Sample configurations
- `config-samples/validation.env.example` – minimal validation-only setup.
- `config-samples/inference.env.example` – inference-only deployment.
- `config-samples/dual-mode.env.example` – dual-mode worker configuration.
- `config-samples/dual-mode.yaml.example` – YAML blueprint for orchestration tools.
- `src/.env.example` – comprehensive baseline with all available options.

#### Documentation
- **README.md** – comprehensive rewrite covering all three modes with usage examples.
- **OPERATIONS.md** – detailed operational guide covering:
  - Architecture diagrams for each mode
  - FedLedger integration patterns
  - Cache management strategies
  - Telemetry and observability setup
  - TLS/certificate management
  - Heartbeat and health monitoring
  - Resource planning guidelines
  - Container orchestration examples (Kubernetes, Docker Compose)
  - Network topology considerations
  - Troubleshooting workflows

#### Dependencies
- Added `fastapi>=0.115.0` for inference API framework.
- Added `uvicorn>=0.30.1` for ASGI server.
- Added `httpx>=0.27.2` for async HTTP client (telemetry).
- Added `aiofiles>=23.2.1` for async file I/O.
- Added `psutil>=5.9.8` for system resource monitoring.
- Added `pyyaml>=6.0.1` for YAML config parsing.
- Added `sse-starlette>=2.1.4` for streaming inference responses.
- Added `cachetools>=5.3.3` for in-memory caching.
- Added `pydantic>=2.7.0` for data validation in API.
- Sorted `requirements.txt` alphabetically for maintainability.

#### Dockerfile enhancements
- Added `WORKER_MODE` environment variable (defaults to `validation`).
- Added `MAX_CACHE_SIZE_GB` environment variable (defaults to `50`).
- Added `TELEMETRY_ENABLED` environment variable (defaults to `false`).
- Exposed port 8000 for inference API in both CPU and GPU images.
- Updated comments to document new environment variables.

### Changed

#### start.sh script
- Complete rewrite with robust argument parsing and validation.
- Process supervision in dual mode – restarts validation on exit code 100 (CUDA errors).
- Unified environment variable precedence: CLI args → env file → env vars → defaults.
- Better error messages and usage documentation.
- Configuration summary logged before launch.
- Graceful shutdown handling with trap for SIGINT/SIGTERM.

#### Documentation structure
- README now has table of contents and clear mode separation.
- Inference API reference with sync and streaming examples.
- Cache management section with best practices.
- TLS termination strategies documented.
- Troubleshooting table expanded with new scenarios.

### Fixed
- `start.sh` executable permission enforced in repository.
- `CACHE_LIMIT_GB` now falls back to `MAX_CACHE_SIZE_GB` for backward compatibility.

### Backward compatibility

All existing validation workflows remain unchanged:
- Single-run validation (`python validate.py validate`) still works.
- Continuous validation loop (`python validate.py loop`) still works.
- Original `start.sh` arguments (`--hf_token`, `--flock_api_key`, `--task_id`) still work.
- CPU and GPU Dockerfiles maintain existing CMD defaults.
- Environment variables (`HF_TOKEN`, `FLOCK_API_KEY`, `TASK_ID`) still work as before.

Users can continue running validation-only workflows without any changes to their deployment scripts.

## [Previous] - Legacy validation-only worker

### Features
- Validation loop with FedLedger assignment polling.
- Support for LoRA and full fine-tuned model validation.
- Automatic cache cleanup between assignments.
- CUDA error recovery (exit code 100).
- GPU type detection and reporting.
- BPC/bPPL metrics calculation.
- Support for multiple base models (Llama, Qwen, Mistral, Phi, etc.).
- FlashAttention support for memory-efficient inference.
