# Inference API Overview

The inference API is implemented with FastAPI and exposes OpenAI-compatible endpoints for synchronous and streaming text generation.

## Endpoints

| Method | Path                  | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/healthz`            | Basic liveness probe.               |
| POST   | `/v1/generate`        | Returns a full completion payload.  |
| POST   | `/v1/generate/stream` | Streams incremental deltas via SSE. |

## Request format

`POST /v1/generate`

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "List two colors."}
  ],
  "temperature": 0.7,
  "max_tokens": 64,
  "stream": false
}
```

Fields match the OpenAI Chat Completions schema. Validation is handled by `server.schemas.ChatCompletionRequest`.

## Response format

```json
{
  "id": "req-123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Two colours are blue and green."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 12,
    "total_tokens": 54
  }
}
```

Streaming responses follow the OpenAI convention of Server-Sent Events (SSE) where each `data:` line contains a `ChatCompletionStreamResponse` fragment. The final message contains `{ "event": "end" }`.

## Configuration

Environment variables (or settings loaded via `.env`) control runtime behaviour:

- `WORKER_MODE`: selects validation/inference/dual mode.
- `INFERENCE_HOST`, `INFERENCE_PORT`: bind address for uvicorn.
- `TELEMETRY_URL`: optional telemetry sink for request metrics.
- `TLS_CERT_PATH`, `TLS_KEY_PATH`: enable TLS termination within the container.
- `API_KEY`: optional API key required for incoming requests.

## Model loading

`server.model_loader.ModelLoader` orchestrates checkpoint loading and leverages `ModelCacheManager` to avoid redundant downloads. By default a lightweight echo model is used for local development; production deployments can inject a custom factory that returns Hugging Face `AutoModelForCausalLM` instances.

## Error handling

- Invalid payloads raise 422 errors with a helpful message from Pydantic.
- Authentication failures raise 401/403 responses when API keys are enabled.
- Internal errors are logged via `loguru` and surfaced as 500 responses.

## Testing strategies

The repository includes tests that cover:

- Schema validation for various request permutations.
- Synchronous generation returning deterministic outputs.
- Streaming generation delivering ordered deltas and a terminating event.
- API key enforcement and TLS option validation.

Use `pytest -q tests/server` to run the suite locally.
