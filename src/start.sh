#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./start.sh --mode <dual|validation|inference> [options]

Required:
  --hf_token <token>          Hugging Face access token (or set HF_TOKEN)
  --flock_api_key <key>       FLock API key (dual/validation only or set FLOCK_API_KEY)
  --task_id <ids>             Comma-separated task IDs (dual/validation only or set TASK_ID)

Common flags:
  --mode <mode>               Operating mode (default: dual or START_MODE env)
  --validation_args_file <path>
                              Validation args JSON (default: validation_config.json.example)

Dual-mode / validation extras:
  --inference_host <host>     Host for embedded inference server (worker mode)
  --inference_port <port>     Port for embedded inference server (worker mode)
  --polling_interval <seconds>
                              Assignment polling cadence
  --telemetry_interval <seconds>
                              Heartbeat cadence (seconds)
  --telemetry_webhook <https_url>
                              HTTPS webhook for telemetry fan-out
  --telemetry_location <label>
                              Location label to include in heartbeat payloads
  --telemetry_worker_id <id>  Override worker identifier in heartbeats
  --telemetry_enabled <true|false>
                              Enable/disable telemetry reporter explicitly
  --lora_only <true|false>    Restrict validation to LoRA submissions
  --auto_clean_cache <true|false>
                              Toggle validation loop cache cleanup

Inference extras:
  --server_host <host>        Host binding for standalone inference server
  --server_port <port>        Port for standalone inference server
  --server_workers <count>    Uvicorn worker count for standalone inference server
  --reload                    Enable autoreload when using run_server.py

Cache / TLS knobs (exported as environment variables):
  --cache_dir <path>
  --cache_max_size_gb <gb>
  --cache_eviction_strategy <LRU|FIFO|SIZE>
  --cache_auto_evict <true|false>
  --tls_enabled <true|false>
  --tls_cert_path <path>
  --tls_key_path <path>
  --require_api_key <true|false>
  --api_key <value>
  --api_key_header <header>
  --reject_non_cached_models <true|false>
  --max_gen_tokens_limit <int>
  --default_max_tokens <int>
  --default_temperature <float>
  --max_context_length <int>

Unrecognised arguments are forwarded to the underlying Python command.
Environment variables such as START_MODE, VALIDATION_ARGS_FILE, TASK_ID, etc. are also honoured.
EOF
}

require_value() {
    local flag="$1"
    local value="$2"
    if [[ -z "$value" || "$value" == --* ]]; then
        printf 'Missing value for %s\n' "$flag" >&2
        usage
        exit 1
    fi
}

run_with_restart() {
    while true; do
        set +e
        "$@"
        local exit_code=$?
        set -e
        if [[ $exit_code -eq 100 ]]; then
            printf 'CUDA error detected, restarting the process...\n' >&2
            continue
        elif [[ $exit_code -ne 0 ]]; then
            printf 'Process exited with code %s\n' "$exit_code" >&2
            return $exit_code
        fi
        break
    done
}

MODE="${START_MODE:-dual}"
VALIDATION_ARGS_FILE="${VALIDATION_ARGS_FILE:-validation_config.json.example}"

HF_TOKEN="${HF_TOKEN:-}"
FLOCK_API_KEY="${FLOCK_API_KEY:-}"
TASK_ID="${TASK_ID:-}"
INFERENCE_HOST_ENV="${INFERENCE_HOST:-}"
INFERENCE_PORT_ENV="${INFERENCE_PORT:-}"
POLLING_INTERVAL_ENV="${POLLING_INTERVAL:-}"
LORA_ONLY_ENV="${LORA_ONLY:-}"
AUTO_CLEAN_CACHE_ENV="${AUTO_CLEAN_CACHE:-}"

WORKER_ARGS=()
LOOP_ARGS=()
INFERENCE_ARGS=()
EXTRA_ARGS=()

set_inference_host=false
set_inference_port=false
set_polling_interval=false
set_lora_only=false
set_auto_clean_cache=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --mode)
            require_value "$1" "${2:-}"
            MODE="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --hf_token)
            require_value "$1" "${2:-}"
            HF_TOKEN="$2"
            export HF_TOKEN
            shift 2
            ;;
        --flock_api_key)
            require_value "$1" "${2:-}"
            FLOCK_API_KEY="$2"
            export FLOCK_API_KEY
            shift 2
            ;;
        --task_id)
            require_value "$1" "${2:-}"
            TASK_ID="$2"
            shift 2
            ;;
        --validation_args_file)
            require_value "$1" "${2:-}"
            VALIDATION_ARGS_FILE="$2"
            shift 2
            ;;
        --inference_host)
            require_value "$1" "${2:-}"
            INFERENCE_HOST_ENV="$2"
            set_inference_host=true
            WORKER_ARGS+=("--inference_host" "$2")
            shift 2
            ;;
        --inference_port)
            require_value "$1" "${2:-}"
            INFERENCE_PORT_ENV="$2"
            set_inference_port=true
            WORKER_ARGS+=("--inference_port" "$2")
            shift 2
            ;;
        --polling_interval)
            require_value "$1" "${2:-}"
            POLLING_INTERVAL_ENV="$2"
            set_polling_interval=true
            WORKER_ARGS+=("--polling_interval" "$2")
            shift 2
            ;;
        --telemetry_interval)
            require_value "$1" "${2:-}"
            WORKER_ARGS+=("--telemetry_interval" "$2")
            shift 2
            ;;
        --telemetry_webhook)
            require_value "$1" "${2:-}"
            export TELEMETRY_WEBHOOK_URL="$2"
            WORKER_ARGS+=("--telemetry_webhook" "$2")
            shift 2
            ;;
        --telemetry_location)
            require_value "$1" "${2:-}"
            export TELEMETRY_LOCATION="$2"
            WORKER_ARGS+=("--telemetry_location" "$2")
            shift 2
            ;;
        --telemetry_worker_id)
            require_value "$1" "${2:-}"
            export TELEMETRY_WORKER_ID="$2"
            WORKER_ARGS+=("--telemetry_worker_id" "$2")
            shift 2
            ;;
        --telemetry_enabled)
            require_value "$1" "${2:-}"
            export TELEMETRY_ENABLED="$2"
            shift 2
            ;;
        --lora_only)
            require_value "$1" "${2:-}"
            LORA_ONLY_ENV="$2"
            set_lora_only=true
            WORKER_ARGS+=("--lora_only" "$2")
            LOOP_ARGS+=("--lora_only" "$2")
            shift 2
            ;;
        --auto_clean_cache)
            require_value "$1" "${2:-}"
            AUTO_CLEAN_CACHE_ENV="$2"
            set_auto_clean_cache=true
            LOOP_ARGS+=("--auto_clean_cache" "$2")
            shift 2
            ;;
        --server_host)
            require_value "$1" "${2:-}"
            export SERVER_HOST="$2"
            INFERENCE_ARGS+=("--host" "$2")
            shift 2
            ;;
        --server_port)
            require_value "$1" "${2:-}"
            export SERVER_PORT="$2"
            INFERENCE_ARGS+=("--port" "$2")
            shift 2
            ;;
        --server_workers)
            require_value "$1" "${2:-}"
            export SERVER_WORKERS="$2"
            INFERENCE_ARGS+=("--workers" "$2")
            shift 2
            ;;
        --reload)
            INFERENCE_ARGS+=("--reload")
            shift 1
            ;;
        --api_key)
            require_value "$1" "${2:-}"
            export API_KEY="$2"
            shift 2
            ;;
        --api_key_header)
            require_value "$1" "${2:-}"
            export API_KEY_HEADER="$2"
            shift 2
            ;;
        --require_api_key)
            require_value "$1" "${2:-}"
            export REQUIRE_API_KEY="$2"
            shift 2
            ;;
        --tls_enabled)
            require_value "$1" "${2:-}"
            export TLS_ENABLED="$2"
            shift 2
            ;;
        --tls_cert_path)
            require_value "$1" "${2:-}"
            export TLS_CERT_PATH="$2"
            shift 2
            ;;
        --tls_key_path)
            require_value "$1" "${2:-}"
            export TLS_KEY_PATH="$2"
            shift 2
            ;;
        --cache_dir)
            require_value "$1" "${2:-}"
            export CACHE_DIR="$2"
            shift 2
            ;;
        --cache_max_size_gb)
            require_value "$1" "${2:-}"
            export CACHE_MAX_SIZE_GB="$2"
            shift 2
            ;;
        --cache_eviction_strategy)
            require_value "$1" "${2:-}"
            export CACHE_EVICTION_STRATEGY="$2"
            shift 2
            ;;
        --cache_auto_evict)
            require_value "$1" "${2:-}"
            export CACHE_AUTO_EVICT="$2"
            shift 2
            ;;
        --reject_non_cached_models)
            require_value "$1" "${2:-}"
            export REJECT_NON_CACHED_MODELS="$2"
            shift 2
            ;;
        --max_gen_tokens_limit)
            require_value "$1" "${2:-}"
            export MAX_GEN_TOKENS_LIMIT="$2"
            shift 2
            ;;
        --default_max_tokens)
            require_value "$1" "${2:-}"
            export DEFAULT_MAX_TOKENS="$2"
            shift 2
            ;;
        --default_temperature)
            require_value "$1" "${2:-}"
            export DEFAULT_TEMPERATURE="$2"
            shift 2
            ;;
        --max_context_length)
            require_value "$1" "${2:-}"
            export MAX_CONTEXT_LENGTH="$2"
            shift 2
            ;;
        *)
            if [[ -n "${2:-}" && "$2" != --* ]]; then
                EXTRA_ARGS+=("$1" "$2")
                shift 2
            else
                EXTRA_ARGS+=("$1")
                shift 1
            fi
            ;;
    esac
done

if ! $set_inference_host && [[ -n "$INFERENCE_HOST_ENV" ]]; then
    WORKER_ARGS+=("--inference_host" "$INFERENCE_HOST_ENV")
fi
if ! $set_inference_port && [[ -n "$INFERENCE_PORT_ENV" ]]; then
    WORKER_ARGS+=("--inference_port" "$INFERENCE_PORT_ENV")
fi
if ! $set_polling_interval && [[ -n "$POLLING_INTERVAL_ENV" ]]; then
    WORKER_ARGS+=("--polling_interval" "$POLLING_INTERVAL_ENV")
fi
if ! $set_lora_only && [[ -n "$LORA_ONLY_ENV" ]]; then
    WORKER_ARGS+=("--lora_only" "$LORA_ONLY_ENV")
    LOOP_ARGS+=("--lora_only" "$LORA_ONLY_ENV")
fi
if ! $set_auto_clean_cache && [[ -n "$AUTO_CLEAN_CACHE_ENV" ]]; then
    LOOP_ARGS+=("--auto_clean_cache" "$AUTO_CLEAN_CACHE_ENV")
fi

if [[ -z "$HF_TOKEN" ]]; then
    printf 'HF_TOKEN must be provided via --hf_token or environment variable.\n' >&2
    exit 1
fi

case "$MODE" in
    dual)
        if [[ -z "$FLOCK_API_KEY" ]]; then
            printf 'FLOCK_API_KEY is required for dual mode.\n' >&2
            exit 1
        fi
        if [[ -z "$TASK_ID" ]]; then
            printf 'task_id is required for dual mode.\n' >&2
            exit 1
        fi
        run_with_restart python validate.py worker \
            --task_id "$TASK_ID" \
            --validation_args_file "$VALIDATION_ARGS_FILE" \
            "${WORKER_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    validation)
        if [[ -z "$FLOCK_API_KEY" ]]; then
            printf 'FLOCK_API_KEY is required for validation mode.\n' >&2
            exit 1
        fi
        if [[ -z "$TASK_ID" ]]; then
            printf 'task_id is required for validation mode.\n' >&2
            exit 1
        fi
        run_with_restart python validate.py loop \
            --task_id "$TASK_ID" \
            --validation_args_file "$VALIDATION_ARGS_FILE" \
            "${LOOP_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    inference)
        run_with_restart python server/run_server.py \
            "${INFERENCE_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    *)
        printf 'Unknown mode: %s\n' "$MODE" >&2
        usage
        exit 1
        ;;
esac
