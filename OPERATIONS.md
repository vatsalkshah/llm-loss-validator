# Operations Guide

This guide covers operational considerations for deploying the llm-loss-validator in production environments across validation-only, inference-only, and dual-mode configurations.

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [FedLedger integration](#fedledger-integration)
3. [Cache management strategies](#cache-management-strategies)
4. [Telemetry and observability](#telemetry-and-observability)
5. [TLS and certificate management](#tls-and-certificate-management)
6. [Heartbeat and health monitoring](#heartbeat-and-health-monitoring)
7. [Resource planning](#resource-planning)
8. [Container orchestration](#container-orchestration)
9. [Network topology](#network-topology)
10. [Troubleshooting workflows](#troubleshooting-workflows)

## Architecture overview

### Validation-only mode

```
┌────────────────────────────────────────────────┐
│         Validation Worker Container            │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  start.sh (orchestrator)                 │ │
│  │                                          │ │
│  │  ┌────────────────────────────────────┐ │ │
│  │  │  validate.py loop                  │ │ │
│  │  │  ─────────────────────────────────  │ │ │
│  │  │  • Request assignments             │ │ │
│  │  │  • Download eval datasets          │ │ │
│  │  │  • Load model + tokenizer          │ │ │
│  │  │  • Run Trainer.evaluate()          │ │ │
│  │  │  • Submit loss metrics             │ │ │
│  │  │  • Clean cache (optional)          │ │ │
│  │  └────────────────────────────────────┘ │ │
│  │                 ↓                         │ │
│  │  ┌────────────────────────────────────┐ │ │
│  │  │  client/fed_ledger.py              │ │ │
│  │  │  (FLock API client)                │ │ │
│  │  └────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  /data/huggingface-cache (volume mount)        │
└────────────────────────────────────────────────┘
                 ↓
        FedLedger REST API
   https://fed-ledger-prod.flock.io
```

### Dual-mode architecture

```
┌───────────────────────────────────────────────────────────────┐
│         Dual-mode Worker Container                            │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  start.sh (process supervisor)                          │ │
│  │                                                         │ │
│  │  ┌────────────────────┐      ┌────────────────────┐   │ │
│  │  │ validate.py loop   │      │ Inference API      │   │ │
│  │  │  (background)      │      │  (uvicorn + ASGI)  │   │ │
│  │  │                    │      │                    │   │ │
│  │  │  ↓                 │      │  ← HTTP requests   │   │ │
│  │  │  FedLedger API     │      │    from clients    │   │ │
│  │  │  (polling)         │      │                    │   │ │
│  │  └────────────────────┘      └────────────────────┘   │ │
│  │          ↓                              ↓              │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │     Shared Hugging Face Cache                    │ │ │
│  │  │     + PEFT adapter cache                         │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │          ↓                                             │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │     Telemetry exporter (optional)                │ │ │
│  │  │     ↓ GPU stats, cache usage, request latencies  │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Port 8000 → Inference API (exposed to clients)              │
└───────────────────────────────────────────────────────────────┘
                 ↓                                ↓
        FedLedger API                     External telemetry
  (assignment lifecycle)                  (metrics ingestion)
```

## FedLedger integration

### Assignment lifecycle

The validation loop interacts with FedLedger through `client/fed_ledger.py`:

1. **Request validation assignment** – `POST /tasks/request-validation-assignment/{task_id}`
2. **Submit validation result** – `POST /tasks/update-validation-assignment/{assignment_id}` with `status=completed` and loss metric.
3. **Mark assignment as failed** – same endpoint but with `status=failed`.

The API enforces a rate limit (1 request per 3 minutes per task). The loop sleeps between polling cycles (configured via `TIME_SLEEP`, default 600s) and handles rate-limit responses gracefully.

### Authentication

`FLOCK_API_KEY` is sent in the `flock-api-key` HTTP header. Store this securely—do not hardcode it in images. Use Kubernetes Secrets, Docker Swarm secrets, or environment injection at runtime.

### Error handling strategies

- **429 (rate limit)** – the loop sleeps for the remainder of the 3-minute window and retries.
- **500/503 (server error)** – after a brief delay the loop retries. Repeated failures trigger `mark_assignment_as_failed`.
- **400/404 (bad request / not found)** – assignment is marked as failed immediately.
- **CUDA errors (exit code 100)** – the outer `start.sh` loop restarts the validation script automatically. Useful when GPU state becomes corrupted after OOM events.

## Cache management strategies

### Shared cache for dual-mode

When running in dual mode, the validation loop and inference API share the same Hugging Face cache directory. Benefits:

- Download base models once; both services use the same checkpoint.
- LoRA adapters fetched during validation remain available for inference inference.
- Reduced disk I/O and storage footprint.

### Cache limits and eviction

Set `CACHE_LIMIT_GB` (or `CACHE_LIMIT_BYTES`) to cap the on-disk cache:

```bash
CACHE_LIMIT_GB=60 ./start.sh --mode dual ...
```

The launcher converts this into bytes. Downstream components (not yet implemented in this repository) would monitor `CACHE_DIR` size and evict least-recently-used snapshots.

### Auto-clean policy

`AUTO_CLEAN_CACHE=true` (default) instructs the loop to remove model directories not listed in `SUPPORTED_BASE_MODELS` between assignments:

```python
# core/constant.py
SUPPORTED_BASE_MODELS = [
    "Qwen/Qwen1.5-1.8B-Chat",
    "meta-llama/Llama-2-7b-chat-hf",
    ...
]
```

Set `AUTO_CLEAN_CACHE=false` if you want to preserve all downloaded checkpoints indefinitely (requires sufficient disk space).

### Persistent volumes

Mount a persistent volume at `/data/huggingface-cache` (or whatever you configure). Without persistence:

- Each container restart downloads fresh weights → high bandwidth usage.
- Startup times increase significantly.

Docker example:

```bash
docker run -v /mnt/data/hf-cache:/data/huggingface-cache \
  -e CACHE_DIR=/data/huggingface-cache \
  ...
```

Kubernetes example:

```yaml
volumeMounts:
  - name: hf-cache
    mountPath: /data/huggingface-cache
volumes:
  - name: hf-cache
    persistentVolumeClaim:
      claimName: huggingface-cache-pvc
```

## Telemetry and observability

### Telemetry export

Set `TELEMETRY_URL` to enable metric export (inference latencies, GPU stats, cache usage):

```bash
TELEMETRY_URL=https://telemetry.example.com/ingest ./start.sh --mode dual ...
```

The launcher exports `TELEMETRY_ENABLED=true` so application code can push JSON payloads to the endpoint at the end of each inference request or validation run.

### TLS client authentication for telemetry

If your telemetry endpoint requires mTLS:

```bash
./start.sh \
  --telemetry-url https://secure-telemetry.example.com/ingest \
  --telemetry-cert-path /secrets/worker-telemetry.crt \
  --telemetry-key-path /secrets/worker-telemetry.key \
  --telemetry-ca-path /etc/ssl/certs/root_ca.pem \
  ...
```

The launcher sets environment variables; your telemetry client should read:

- `TELEMETRY_TLS_CERT_PATH`
- `TELEMETRY_TLS_KEY_PATH`
- `TELEMETRY_TLS_CA_PATH`

Example Python snippet:

```python
import httpx
import os

cert_path = os.getenv("TELEMETRY_TLS_CERT_PATH")
key_path = os.getenv("TELEMETRY_TLS_KEY_PATH")
ca_path = os.getenv("TELEMETRY_TLS_CA_PATH")

if cert_path and key_path:
    client = httpx.Client(
        cert=(cert_path, key_path),
        verify=ca_path if ca_path else True,
    )
```

### Metrics to collect

- **Validation metrics** – assignment ID, model name, eval loss, token count, GPU type, runtime duration.
- **Inference metrics** – request ID, latency (p50/p95/p99), tokens/second, finish reason, cache hit rate.
- **Resource metrics** – GPU memory utilization, GPU temperature, cache directory size, CPU/RAM usage.

## TLS and certificate management

### Scenario 1: External TLS termination (recommended)

Deploy an ingress proxy (nginx, Traefik, AWS ALB) in front of the worker:

```
Internet → Ingress (TLS) → Worker (HTTP on port 8000)
```

The worker configuration remains simple:

```bash
TLS_ENABLED=false
INFERENCE_PORT=8000
```

This approach separates concerns: the proxy handles cert rotation, cipher suite configuration, and HTTP/2 negotiation.

### Scenario 2: Direct TLS termination at the worker

Mount certificates into the container and configure `uvicorn` to serve HTTPS:

```bash
./start.sh \
  --mode inference \
  --inference-module src.inference.server:app \
  --inference-extra "--ssl-keyfile /secrets/server.key --ssl-certfile /secrets/server.crt" \
  ...
```

Set:

```bash
TLS_ENABLED=true
TLS_CERT_PATH=/secrets/server.crt
TLS_KEY_PATH=/secrets/server.key
```

Use this pattern when:

- Running behind a TCP load balancer (no TLS termination).
- mTLS between client and worker is required.
- Compliance mandates end-to-end encryption.

### Certificate rotation

When certificates expire:

1. Update the secret (Kubernetes Secret, Docker Swarm secret, volume mount).
2. Restart the worker container. The launcher reads paths at startup and passes them to `uvicorn`.

For zero-downtime rotation, use a sidecar container to watch the secret and signal the main process to reload (requires custom signal handling in the ASGI app).

## Heartbeat and health monitoring

### Heartbeat reports

Optional periodic status pings to FedLedger:

```bash
HEARTBEAT_ENDPOINT=https://fed-ledger-prod.flock.io/api/v1/worker-heartbeat
HEARTBEAT_INTERVAL=60  # seconds
```

Heartbeat payloads might include:

- Worker mode (validation/inference/dual)
- Current model loaded
- GPU availability
- Cache usage stats

This allows the platform to detect stale workers and reassign tasks accordingly.

### Health check endpoints

Implement a `/health` endpoint in your inference ASGI app:

```python
@app.get("/health")
def health_check():
    return {"status": "ok", "mode": os.getenv("WORKER_MODE")}
```

Configure your orchestrator to probe this endpoint:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  periodSeconds: 30
  failureThreshold: 3
```

### Readiness vs liveness

- **Liveness** – is the container running? Restart if `/health` fails repeatedly.
- **Readiness** – is the service ready to accept traffic? Only route traffic if the model is loaded and the cache is accessible.

Example readiness check:

```python
@app.get("/ready")
def readiness_check():
    if model_loaded and cache_writable:
        return {"ready": True}
    else:
        raise HTTPException(status_code=503, detail="Not ready")
```

## Resource planning

### CPU-based validation

- **vCPUs** – 4–8 cores recommended. Transformers DataLoader uses multiple workers.
- **RAM** – 16–32 GiB depending on model size.
- **Disk** – 50–200 GiB for cache. Use SSD for faster I/O.
- **Network** – 100 Mbps+ for Hugging Face Hub downloads.

### GPU-based validation/inference

- **GPU memory** – 16 GiB minimum for 7B models. 24–40 GiB for 13B–30B models.
- **vCPUs** – 8–16 cores.
- **RAM** – 32–64 GiB.
- **Disk** – 100–500 GiB (models + cache + adapter snapshots).
- **Network** – 1 Gbps+ recommended for initial downloads.

### Dual-mode resource sharing

Running both validation and inference on the same GPU requires careful scheduling:

- Validation runs in a blocking loop; inference requests queue during validation.
- Consider time-slicing GPUs (NVIDIA MIG, CUDA MPS) or dedicating one GPU per mode.
- Monitor GPU memory to avoid OOM (out-of-memory) events.

## Container orchestration

### Kubernetes deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flock-validator-dual
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flock-validator
  template:
    metadata:
      labels:
        app: flock-validator
    spec:
      containers:
      - name: worker
        image: ghcr.io/yourorg/flock-validator:gpu
        command: ["/bin/bash", "/app/src/start.sh"]
        args:
          - "--mode"
          - "dual"
          - "--env-file"
          - "/config/worker.env"
          - "--inference-module"
          - "src.inference.server:app"
          - "--cache-dir"
          - "/data/hf-cache"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: hf-cache
          mountPath: /data/hf-cache
        - name: tls-secrets
          mountPath: /secrets
          readOnly: true
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
      volumes:
      - name: config
        configMap:
          name: flock-validator-config
      - name: hf-cache
        persistentVolumeClaim:
          claimName: huggingface-cache-pvc
      - name: tls-secrets
        secret:
          secretName: tls-certificates
```

### Docker Compose

```yaml
version: '3.8'
services:
  validator:
    image: flock-validator:gpu
    runtime: nvidia
    environment:
      - WORKER_MODE=dual
      - HF_TOKEN=${HF_TOKEN}
      - FLOCK_API_KEY=${FLOCK_API_KEY}
      - TASK_ID=8,9
      - INFERENCE_MODULE=src.inference.server:app
      - CACHE_DIR=/data/hf-cache
      - CACHE_LIMIT_GB=60
      - TELEMETRY_URL=https://telemetry.example.com/ingest
    volumes:
      - hf-cache:/data/hf-cache
      - ./secrets:/secrets:ro
    ports:
      - "8000:8000"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  hf-cache:
```

## Network topology

### Firewall rules

- **Validation-only** – allow outbound HTTPS to `fed-ledger-prod.flock.io` and `huggingface.co`.
- **Inference-only** – allow inbound HTTP(S) on port 8000 (or configured port). Allow outbound to Hugging Face Hub.
- **Dual-mode** – combine both rulesets.

### Service mesh integration

If using Istio or Linkerd:

- Annotate the pod to inject the sidecar proxy.
- Configure mTLS for service-to-service communication.
- Use VirtualServices to route `/v1/generate*` to the inference API.

### DNS and service discovery

For multiple workers:

- Deploy a service (Kubernetes Service, Consul service) that load-balances across pods.
- Use sticky sessions if your inference API maintains per-request state.

## Troubleshooting workflows

### Validation loop exits immediately

**Symptoms:**

- Container restarts frequently.
- Logs show "Rate limit reached" or "No task submissions available".

**Checks:**

1. Verify `TASK_ID` points to active tasks on the FLock platform.
2. Confirm the machine clock is reasonably accurate (NTP sync).
3. Inspect logs for FedLedger HTTP status codes (401 = bad API key, 404 = task not found).
4. Ensure `FLOCK_API_KEY` and `HF_TOKEN` are correctly set.

### Inference API returns 5xx errors

**Symptoms:**

- Clients receive `500 Internal Server Error` or `503 Service Unavailable`.

**Checks:**

1. Check ASGI logs for Python exceptions (model not loaded, CUDA OOM).
2. Verify GPU memory is sufficient (`nvidia-smi`).
3. Ensure the cache directory is writable and has free space.
4. Confirm the model/tokenizer paths are correct.

### Cache fills up rapidly

**Symptoms:**

- Disk usage grows unbounded.
- "No space left on device" errors in logs.

**Fixes:**

1. Increase `CACHE_LIMIT_GB` or provision a larger volume.
2. Enable `AUTO_CLEAN_CACHE=true`.
3. Manually prune old snapshots: `rm -rf /data/hf-cache/models--*`.
4. Configure the cache manager (implementation-dependent) to evict LRU entries.

### Telemetry not appearing in backend

**Symptoms:**

- No metrics visible in your observability platform.

**Checks:**

1. Verify `TELEMETRY_URL` is correct and reachable (`curl` from inside the container).
2. Check TLS cert/key paths if using mTLS.
3. Inspect application logs for HTTP errors when posting telemetry.
4. Confirm firewall rules allow outbound HTTPS to the telemetry endpoint.

### TLS handshake failures

**Symptoms:**

- Clients see SSL errors when connecting to inference API.
- Telemetry client logs certificate validation errors.

**Fixes:**

1. Verify certificate and key files exist at the specified paths.
2. Check file permissions (should be readable by the container user).
3. Confirm the certificate chain is complete (intermediate + root CA).
4. Use `openssl s_client -connect <host>:<port>` to debug handshake.
5. Ensure cert and key match (`openssl x509 -noout -modulus -in cert.pem | openssl md5`).

### GPU not detected or CUDA errors

**Symptoms:**

- "CUDA not available" warnings.
- Exit code 100 (CUDA runtime error).

**Checks:**

1. Ensure NVIDIA drivers are installed on the host.
2. Confirm Docker runtime is set to `nvidia` (`docker run --gpus all ...`).
3. Set `CUDA_VISIBLE_DEVICES` to explicitly select a GPU.
4. Restart the container to clear corrupted GPU state.
5. Check `nvidia-smi` on the host to verify GPU health.

### Validation loop and inference API both crash in dual mode

**Symptoms:**

- Both processes terminate shortly after startup.
- Logs show resource exhaustion.

**Fixes:**

1. Increase memory allocation (CPU RAM or GPU VRAM).
2. Reduce batch sizes in `validation_config.json.example`.
3. Consider splitting workloads across separate containers.
4. Monitor resource usage with `docker stats` or `kubectl top pods`.

## Additional resources

- [Hugging Face Hub documentation](https://huggingface.co/docs/hub/index)
- [PyTorch Transformers documentation](https://huggingface.co/docs/transformers/index)
- [Uvicorn deployment guide](https://www.uvicorn.org/deployment/)
- [FastAPI production best practices](https://fastapi.tiangolo.com/deployment/)
- [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
