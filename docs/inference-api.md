# Inference API Server

## Overview

The inference API server provides an OpenAI-compatible HTTP API for serving language models with streaming support. It integrates with the existing model cache manager to efficiently load and serve models.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions API
- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming
- **Model Caching**: Integration with ModelCacheManager for efficient model loading
- **Authentication**: Optional API key authentication
- **TLS Support**: Optional HTTPS/TLS configuration
- **Request Logging**: Configurable logging for requests and responses
- **Auto Template Detection**: Automatically detects chat templates for popular models

## Quick Start

### Basic Usage

```bash
# Set environment variables
export HF_TOKEN=your_hf_token
export API_KEY=your_api_key  # Optional

# Start the server
cd src
python -m server.main
```

The server will start on `http://0.0.0.0:8000` by default.

### Using Uvicorn directly

```bash
cd src
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Configure the server using environment variables:

### Server Settings

- `SERVER_HOST`: Host to bind to (default: `0.0.0.0`)
- `SERVER_PORT`: Port to bind to (default: `8000`)
- `SERVER_WORKERS`: Number of worker processes (default: `1`)

### Authentication

- `REQUIRE_API_KEY`: Enable API key authentication (default: `false`)
- `API_KEY`: The API key to validate against
- `API_KEY_HEADER`: Header name for API key (default: `X-API-Key`)

### TLS/SSL

- `TLS_ENABLED`: Enable TLS (default: `false`)
- `TLS_CERT_PATH`: Path to SSL certificate file
- `TLS_KEY_PATH`: Path to SSL private key file

### Logging

- `LOG_REQUESTS`: Log incoming requests (default: `true`)
- `LOG_RESPONSES`: Log outgoing responses (default: `false`)
- `LOG_PAYLOADS`: Log request/response payloads (default: `false`)

### Model Settings

- `HF_TOKEN`: HuggingFace authentication token (required)
- `DEFAULT_MAX_TOKENS`: Default max tokens for generation (default: `512`)
- `DEFAULT_TEMPERATURE`: Default temperature (default: `0.7`)
- `MAX_CONTEXT_LENGTH`: Maximum context length (default: `4096`)
- `MAX_GEN_TOKENS_LIMIT`: Hard limit on max_tokens parameter (default: `2048`)
- `REJECT_NON_CACHED_MODELS`: Reject requests for non-cached models (default: `false`)

### Cache Settings

- `CACHE_DIR`: Cache directory path (default: `~/.cache/huggingface`)
- `CACHE_ENABLED`: Enable cache management (default: `true`)
- `CACHE_MAX_SIZE_GB`: Maximum cache size in GB (default: `100`)
- `CACHE_EVICTION_STRATEGY`: Eviction strategy - LRU, FIFO, SIZE (default: `LRU`)

## API Endpoints

### POST /v1/chat/completions

Create a chat completion.

**Request Body:**

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 1.0,
  "stream": false,
  "stop": null
}
```

**Parameters:**

- `model` (string, required): Model identifier
- `messages` (array, required): List of chat messages
- `temperature` (float, optional): Sampling temperature (0.0-2.0)
- `top_p` (float, optional): Nucleus sampling (0.0-1.0)
- `max_tokens` (integer, optional): Maximum tokens to generate
- `stream` (boolean, optional): Enable streaming (default: false)
- `stop` (string or array, optional): Stop sequences
- `presence_penalty` (float, optional): Presence penalty (-2.0-2.0) [not yet implemented]
- `frequency_penalty` (float, optional): Frequency penalty (-2.0-2.0) [not yet implemented]

**Non-Streaming Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 12,
    "total_tokens": 27
  }
}
```

**Streaming Response:**

Server-Sent Events (SSE) format:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### GET /v1/models

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local",
      "cached": true,
      "loaded": true
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy"
}
```

## Example Usage

### cURL (Non-streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is AI?"}
    ],
    "max_tokens": 100
  }'
```

### cURL (Streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

### Python (OpenAI client)

```python
from openai import OpenAI

# Configure client for local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your_api_key"
)

# Non-streaming
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Python (Requests)

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_api_key"
}

# Non-streaming
payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])

# Streaming
payload["stream"] = True
response = requests.post(url, headers=headers, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = line_str[6:]
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                print(content, end="")
```

## Security

### API Key Authentication

Enable API key authentication:

```bash
export REQUIRE_API_KEY=true
export API_KEY=your_secret_key
```

Clients must include the API key in the request header:

```bash
curl -H "X-API-Key: your_secret_key" ...
```

### TLS/SSL

Enable HTTPS with TLS:

```bash
export TLS_ENABLED=true
export TLS_CERT_PATH=/path/to/cert.pem
export TLS_KEY_PATH=/path/to/key.pem
```

Alternatively, use a reverse proxy (nginx, Caddy) for TLS termination.

## Model Cache Integration

The server integrates with the existing `ModelCacheManager` to:

- Track cached models
- Load models on-demand
- Mark models as active when in use
- Support model eviction policies

When `REJECT_NON_CACHED_MODELS=true`, the server will reject requests for models not in the cache with a 404 error, providing an actionable error message.

## Supported Models

The server automatically detects chat templates for:

- LLaMA 2 and LLaMA 3 models
- Qwen/Qwen2.5 models
- Yi models
- Mistral and Mixtral models
- Gemma models
- Phi-3 and Phi-4 models
- Zephyr models

For other models, a default template is used.

## Testing

Run tests:

```bash
pytest tests/server/
```

## Troubleshooting

### Model Not Found

If you get "Model not found in cache":
1. Ensure the model is downloaded via the cache manager
2. Check `REJECT_NON_CACHED_MODELS` setting
3. Verify model ID is correct

### Out of Memory

If you encounter OOM errors:
1. Reduce `MAX_CONTEXT_LENGTH`
2. Reduce `MAX_GEN_TOKENS_LIMIT`
3. Use smaller models
4. Enable model eviction with `CACHE_AUTO_EVICT=true`

### Streaming Issues

If streaming doesn't work:
1. Ensure client supports SSE
2. Check for buffering in reverse proxies
3. Verify `stream: true` in request

## Development

### Adding New Endpoints

Add endpoints in `src/server/main.py`:

```python
@app.get("/custom/endpoint")
async def custom_endpoint(api_key: Optional[str] = Depends(api_key_header)):
    await verify_api_key(api_key)
    # Implementation
    return {"result": "data"}
```

### Custom Templates

Add templates in `src/core/template.py` or override auto-detection in `src/server/generator.py`.
