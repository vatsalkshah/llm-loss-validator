#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./start.sh [options]

Core options:
  --mode <validation|inference|dual>   Worker mode (defaults to WORKER_MODE env var or "validation")
  --env-file <path>                    Load KEY=VALUE pairs from a dotenv file before launching
  --hf_token <token>                   Hugging Face Hub token (or set HF_TOKEN env var)
  --flock_api_key <key>                FLock API key (required in validation/dual modes)
  --task_id <ids>                      Comma separated task IDs monitored by the validation loop
  --validation_args_file <path>        Validation config JSON passed to validate.py (default: validation_config.json.example)

Validation behaviour:
  --auto_clean_cache <bool>            Toggle cache cleanup between assignments (default: true)
  --lora_only <bool>                   Only accept LoRA submissions (default: true)

Inference behaviour:
  --inference-command "<cmd>"          Full shell command used to launch the inference API
  --inference-module <module:app>      Run uvicorn for the specified ASGI app instead of --inference-command
  --inference-host <host>              Bind host for uvicorn when using --inference-module (default: 0.0.0.0)
  --inference-port <port>              Bind port for the inference API (default: 8000)
  --inference-extra "<args>"           Additional arguments appended to the inference command (repeatable)

Telemetry & heartbeat:
  --telemetry-url <url>                Telemetry ingestion endpoint
  --telemetry-cert-path <path>         Client certificate for telemetry mTLS
  --telemetry-key-path <path>          Client key for telemetry mTLS
  --telemetry-ca-path <path>           CA bundle for telemetry TLS validation
  --heartbeat-endpoint <url>           Override FedLedger heartbeat endpoint
  --heartbeat-interval <seconds>       Override heartbeat cadence (default: 60)

Cache management:
  --cache-dir <path>                   Directory used for Hugging Face cache (sets HF_HOME)
  --cache-limit-bytes <bytes>          Maximum on-disk cache footprint before eviction
  --cache-limit-gb <gigabytes>         Convenience helper to set cache limit using GiB

Additional:
  --help                               Show this message and exit

Examples:
  HF_TOKEN=xxx FLOCK_API_KEY=yyy TASK_ID=8 ./start.sh --mode validation --validation_args_file validation_config.json.example

  HF_TOKEN=xxx ./start.sh --mode inference \
      --inference-module src.inference.server:app \
      --inference-extra "--reload" \
      --telemetry-url https://telemetry.example.com/ingest

  ./start.sh --mode dual --env-file ../config-samples/dual-mode.env.example \
      --inference-module src.inference.server:app

EOF
    exit 1
}

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
CLI_ENV_FILE=""
CLI_MODE=""
CLI_HF_TOKEN=""
CLI_FLOCK_API_KEY=""
CLI_TASK_ID=""
CLI_VALIDATION_ARGS_FILE=""
CLI_AUTO_CLEAN_CACHE=""
CLI_LORA_ONLY=""
CLI_INFERENCE_COMMAND=""
CLI_INFERENCE_MODULE=""
CLI_INFERENCE_HOST=""
CLI_INFERENCE_PORT=""
CLI_CACHE_DIR=""
CLI_CACHE_LIMIT_BYTES=""
CLI_CACHE_LIMIT_GB=""
CLI_TELEMETRY_URL=""
CLI_TELEMETRY_TLS_CERT=""
CLI_TELEMETRY_TLS_KEY=""
CLI_TELEMETRY_TLS_CA=""
CLI_HEARTBEAT_ENDPOINT=""
CLI_HEARTBEAT_INTERVAL=""
declare -a VALIDATION_EXTRA_ARGS=()
declare -a INFERENCE_EXTRA_ARGS=()

missing_value() {
    echo "[start.sh] Missing value for $1" >&2
    usage
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_ENV_FILE="$2"
            shift 2
            ;;
        --mode)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_MODE="$2"
            shift 2
            ;;
        --hf_token)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_HF_TOKEN="$2"
            shift 2
            ;;
        --flock_api_key)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_FLOCK_API_KEY="$2"
            shift 2
            ;;
        --task_id)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_TASK_ID="$2"
            shift 2
            ;;
        --validation_args_file)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_VALIDATION_ARGS_FILE="$2"
            shift 2
            ;;
        --auto_clean_cache)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_AUTO_CLEAN_CACHE="$2"
            shift 2
            ;;
        --lora_only)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_LORA_ONLY="$2"
            shift 2
            ;;
        --inference-command)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_INFERENCE_COMMAND="$2"
            shift 2
            ;;
        --inference-module)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_INFERENCE_MODULE="$2"
            shift 2
            ;;
        --inference-host)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_INFERENCE_HOST="$2"
            shift 2
            ;;
        --inference-port)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_INFERENCE_PORT="$2"
            shift 2
            ;;
        --inference-extra)
            [[ $# -lt 2 ]] && missing_value "$1"
            read -r -a __tmp <<< "$2"
            INFERENCE_EXTRA_ARGS+=("${__tmp[@]}")
            shift 2
            ;;
        --telemetry-url)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_TELEMETRY_URL="$2"
            shift 2
            ;;
        --telemetry-cert-path)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_TELEMETRY_TLS_CERT="$2"
            shift 2
            ;;
        --telemetry-key-path)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_TELEMETRY_TLS_KEY="$2"
            shift 2
            ;;
        --telemetry-ca-path)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_TELEMETRY_TLS_CA="$2"
            shift 2
            ;;
        --heartbeat-endpoint)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_HEARTBEAT_ENDPOINT="$2"
            shift 2
            ;;
        --heartbeat-interval)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_HEARTBEAT_INTERVAL="$2"
            shift 2
            ;;
        --cache-dir)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_CACHE_DIR="$2"
            shift 2
            ;;
        --cache-limit-bytes)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_CACHE_LIMIT_BYTES="$2"
            shift 2
            ;;
        --cache-limit-gb)
            [[ $# -lt 2 ]] && missing_value "$1"
            CLI_CACHE_LIMIT_GB="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [[ $# -gt 1 && "$2" != --* ]]; then
                VALIDATION_EXTRA_ARGS+=("$1" "$2")
                shift 2
            else
                VALIDATION_EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Load environment file if provided
# -----------------------------------------------------------------------------
ENV_FILE=${CLI_ENV_FILE:-${ENV_FILE:-}}
if [[ -n "$ENV_FILE" ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "[start.sh] Environment file not found: $ENV_FILE" >&2
        exit 1
    fi
    echo "[start.sh] Loading environment variables from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# -----------------------------------------------------------------------------
# Resolve configuration with precedence: CLI > env > defaults
# -----------------------------------------------------------------------------
DEFAULT_VALIDATION_ARGS_FILE="validation_config.json.example"
MODE=${CLI_MODE:-${WORKER_MODE:-validation}}
MODE=${MODE,,}
HF_TOKEN=${CLI_HF_TOKEN:-${HF_TOKEN:-}}
FLOCK_API_KEY=${CLI_FLOCK_API_KEY:-${FLOCK_API_KEY:-}}
TASK_ID=${CLI_TASK_ID:-${TASK_ID:-}}
VALIDATION_ARGS_FILE=${CLI_VALIDATION_ARGS_FILE:-${VALIDATION_ARGS_FILE:-$DEFAULT_VALIDATION_ARGS_FILE}}
AUTO_CLEAN_CACHE=${CLI_AUTO_CLEAN_CACHE:-${AUTO_CLEAN_CACHE:-true}}
LORA_ONLY=${CLI_LORA_ONLY:-${LORA_ONLY:-true}}
INFERENCE_COMMAND=${CLI_INFERENCE_COMMAND:-${INFERENCE_COMMAND:-}}
INFERENCE_MODULE=${CLI_INFERENCE_MODULE:-${INFERENCE_MODULE:-}}
INFERENCE_HOST=${CLI_INFERENCE_HOST:-${INFERENCE_HOST:-0.0.0.0}}
INFERENCE_PORT=${CLI_INFERENCE_PORT:-${INFERENCE_PORT:-8000}}
CACHE_DIR=${CLI_CACHE_DIR:-${CACHE_DIR:-${HF_HOME:-}}}
CACHE_LIMIT_BYTES=${CLI_CACHE_LIMIT_BYTES:-${CACHE_LIMIT_BYTES:-}}
CACHE_LIMIT_GB=${CLI_CACHE_LIMIT_GB:-${CACHE_LIMIT_GB:-${MAX_CACHE_SIZE_GB:-}}}
TELEMETRY_URL=${CLI_TELEMETRY_URL:-${TELEMETRY_URL:-}}
TELEMETRY_TLS_CERT=${CLI_TELEMETRY_TLS_CERT:-${TELEMETRY_TLS_CERT:-${TLS_CERT_PATH:-}}}
TELEMETRY_TLS_KEY=${CLI_TELEMETRY_TLS_KEY:-${TELEMETRY_TLS_KEY:-${TLS_KEY_PATH:-}}}
TELEMETRY_TLS_CA=${CLI_TELEMETRY_TLS_CA:-${TELEMETRY_TLS_CA:-${TLS_CA_PATH:-}}}
HEARTBEAT_ENDPOINT=${CLI_HEARTBEAT_ENDPOINT:-${HEARTBEAT_ENDPOINT:-}}
HEARTBEAT_INTERVAL=${CLI_HEARTBEAT_INTERVAL:-${HEARTBEAT_INTERVAL:-60}}

if [[ -z "$HF_TOKEN" ]]; then
    echo "[start.sh] HF_TOKEN must be provided via --hf_token or environment." >&2
    exit 1
fi

if [[ "$MODE" == "validation" || "$MODE" == "dual" ]]; then
    if [[ -z "$FLOCK_API_KEY" ]]; then
        echo "[start.sh] FLOCK_API_KEY must be provided for validation or dual mode." >&2
        exit 1
    fi
    if [[ -z "$TASK_ID" ]]; then
        echo "[start.sh] TASK_ID must be provided for validation or dual mode." >&2
        exit 1
    fi
fi

if [[ "$MODE" == "inference" || "$MODE" == "dual" ]]; then
    if [[ -n "$INFERENCE_COMMAND" && -n "$INFERENCE_MODULE" ]]; then
        echo "[start.sh] Provide either --inference-command or --inference-module, not both." >&2
        exit 1
    fi
    if [[ -z "$INFERENCE_COMMAND" && -z "$INFERENCE_MODULE" ]]; then
        echo "[start.sh] Inference mode requires --inference-command or --inference-module." >&2
        exit 1
    fi
fi

if [[ -n "$CACHE_LIMIT_GB" && -z "$CACHE_LIMIT_BYTES" ]]; then
    if ! CACHE_LIMIT_BYTES=$(python - "$CACHE_LIMIT_GB" <<'PY'
import sys
from decimal import Decimal

try:
    gb = Decimal(sys.argv[1])
except Exception as exc:
    print(f"failed-to-parse:{exc}", file=sys.stderr)
    sys.exit(1)

print(int(gb * (1024 ** 3)))
PY
    ); then
        echo "[start.sh] Failed to convert --cache-limit-gb value: $CACHE_LIMIT_GB" >&2
        exit 1
    fi
fi

if [[ "$MODE" != "validation" && "$MODE" != "inference" && "$MODE" != "dual" ]]; then
    echo "[start.sh] Unsupported mode: $MODE" >&2
    usage
fi

# -----------------------------------------------------------------------------
# Export environment for downstream components
# -----------------------------------------------------------------------------
export HF_TOKEN
if [[ -n "$FLOCK_API_KEY" ]]; then export FLOCK_API_KEY; fi
export WORKER_MODE="$MODE"
export AUTO_CLEAN_CACHE
export LORA_ONLY
export INFERENCE_HOST
export INFERENCE_PORT
if [[ -n "$CACHE_DIR" ]]; then
    export HF_HOME="$CACHE_DIR"
    export CACHE_DIR
fi
if [[ -n "$CACHE_LIMIT_BYTES" ]]; then export CACHE_LIMIT_BYTES; fi
if [[ -n "$TELEMETRY_URL" ]]; then
    export TELEMETRY_URL
    export TELEMETRY_ENABLED="true"
else
    export TELEMETRY_ENABLED="${TELEMETRY_ENABLED:-false}"
fi
if [[ -n "$TELEMETRY_TLS_CERT" ]]; then export TELEMETRY_TLS_CERT_PATH="$TELEMETRY_TLS_CERT"; fi
if [[ -n "$TELEMETRY_TLS_KEY" ]]; then export TELEMETRY_TLS_KEY_PATH="$TELEMETRY_TLS_KEY"; fi
if [[ -n "$TELEMETRY_TLS_CA" ]]; then export TELEMETRY_TLS_CA_PATH="$TELEMETRY_TLS_CA"; fi
if [[ -n "$HEARTBEAT_ENDPOINT" ]]; then export HEARTBEAT_ENDPOINT; fi
if [[ -n "$HEARTBEAT_INTERVAL" ]]; then export HEARTBEAT_INTERVAL; fi

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
declare -a VALIDATION_CMD=()
declare -a INFERENCE_CMD=()

build_validation_command() {
    VALIDATION_CMD=(
        python validate.py loop
        --task_id "$TASK_ID"
        --validation_args_file "$VALIDATION_ARGS_FILE"
        --auto_clean_cache "$AUTO_CLEAN_CACHE"
        --lora_only "$LORA_ONLY"
    )
    if [[ ${#VALIDATION_EXTRA_ARGS[@]} -gt 0 ]]; then
        VALIDATION_CMD+=("${VALIDATION_EXTRA_ARGS[@]}")
    fi
}

build_inference_command() {
    INFERENCE_CMD=()
    if [[ -n "$INFERENCE_COMMAND" ]]; then
        INFERENCE_CMD=(bash -lc "$INFERENCE_COMMAND")
    else
        INFERENCE_CMD=(uvicorn "$INFERENCE_MODULE" "--host" "$INFERENCE_HOST" "--port" "$INFERENCE_PORT")
        if [[ ${#INFERENCE_EXTRA_ARGS[@]} -gt 0 ]]; then
            INFERENCE_CMD+=("${INFERENCE_EXTRA_ARGS[@]}")
        fi
    fi
}

run_validation_loop() {
    build_validation_command
    while true; do
        "${VALIDATION_CMD[@]}"
        local exit_code=$?
        if [[ $exit_code -eq 100 ]]; then
            echo "[start.sh] CUDA error detected, restarting validation loop..."
            continue
        elif [[ $exit_code -ne 0 ]]; then
            echo "[start.sh] Validation loop exited with status $exit_code" >&2
            return $exit_code
        fi
        break
    done
    return 0
}

run_inference_foreground() {
    build_inference_command
    echo "[start.sh] Starting inference server: ${INFERENCE_CMD[*]}"
    exec "${INFERENCE_CMD[@]}"
}

start_inference_background() {
    build_inference_command
    echo "[start.sh] Starting inference server in background: ${INFERENCE_CMD[*]}"
    "${INFERENCE_CMD[@]}" &
    INFERENCE_PID=$!
}

stop_process_if_running() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

log_summary() {
    echo "------------------------------------------------------------"
    echo "[start.sh] Worker mode          : $MODE"
    if [[ "$MODE" != "inference" ]]; then
        echo "[start.sh] Task IDs             : $TASK_ID"
        echo "[start.sh] Validation config    : $VALIDATION_ARGS_FILE"
        echo "[start.sh] Auto clean cache     : $AUTO_CLEAN_CACHE"
        echo "[start.sh] LoRA only            : $LORA_ONLY"
    fi
    if [[ "$MODE" != "validation" ]]; then
        if [[ -n "$INFERENCE_COMMAND" ]]; then
            echo "[start.sh] Inference command    : $INFERENCE_COMMAND"
        else
            echo "[start.sh] Inference module     : $INFERENCE_MODULE"
            echo "[start.sh] Inference bind       : $INFERENCE_HOST:$INFERENCE_PORT"
        fi
    fi
    if [[ -n "$CACHE_DIR" ]]; then
        echo "[start.sh] Cache directory      : $CACHE_DIR"
    fi
    if [[ -n "$CACHE_LIMIT_BYTES" ]]; then
        echo "[start.sh] Cache limit (bytes)  : $CACHE_LIMIT_BYTES"
    fi
    if [[ -n "$TELEMETRY_URL" ]]; then
        echo "[start.sh] Telemetry endpoint   : $TELEMETRY_URL"
    fi
    if [[ -n "$HEARTBEAT_ENDPOINT" ]]; then
        echo "[start.sh] Heartbeat endpoint   : $HEARTBEAT_ENDPOINT"
    fi
    echo "------------------------------------------------------------"
}

# -----------------------------------------------------------------------------
# Launch according to mode
# -----------------------------------------------------------------------------
log_summary

case "$MODE" in
    validation)
        run_validation_loop
        ;;
    inference)
        run_inference_foreground
        ;;
    dual)
        start_inference_background
        trap 'stop_process_if_running "$INFERENCE_PID"; stop_process_if_running "$VALIDATION_PID"' EXIT SIGINT SIGTERM
        run_validation_loop &
        VALIDATION_PID=$!
        while true; do
            if wait -n "$VALIDATION_PID" "$INFERENCE_PID"; then
                EXIT_STATUS=$?
            else
                EXIT_STATUS=$?
            fi
            if ! kill -0 "$VALIDATION_PID" 2>/dev/null; then
                echo "[start.sh] Validation loop terminated (code $EXIT_STATUS)"
                stop_process_if_running "$INFERENCE_PID"
                exit $EXIT_STATUS
            fi
            if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
                echo "[start.sh] Inference server terminated (code $EXIT_STATUS)"
                stop_process_if_running "$VALIDATION_PID"
                exit $EXIT_STATUS
            fi
        done
        ;;
    *)
        echo "[start.sh] Unknown mode: $MODE" >&2
        usage
        ;;
esac
