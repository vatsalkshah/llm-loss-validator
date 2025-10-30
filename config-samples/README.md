# Configuration Templates

This directory contains sample configuration files for different deployment scenarios of the llm-loss-validator worker.

## Files

### validation.env.example

Minimal configuration for running the worker in **validation-only mode** (the legacy/default behavior).

**Use when:**
- You only want to participate in FLock's validation network.
- You don't need to serve inference requests.
- You want the simplest setup.

**How to use:**
```bash
cp validation.env.example .env
# Edit .env with your actual tokens and settings
cd ../src
./start.sh --env-file ../config-samples/.env
```

### inference.env.example

Configuration for running the worker in **inference-only mode** to serve model predictions without participating in validation.

**Use when:**
- You want to host a public or private inference API.
- You don't want to validate models for the FLock network.
- You need to serve predictions to applications.

**How to use:**
```bash
cp inference.env.example .env
# Edit .env with your HF token and inference settings
cd ../src
./start.sh --env-file ../config-samples/.env
```

### dual-mode.env.example

Configuration for running the worker in **dual mode** where both validation loop and inference API run simultaneously.

**Use when:**
- You want to maximize hardware utilization.
- You have sufficient GPU memory to handle both workloads.
- You want to participate in validation while also serving inference.

**How to use:**
```bash
cp dual-mode.env.example .env
# Edit .env with all required tokens and settings
cd ../src
./start.sh --env-file ../config-samples/.env
```

### dual-mode.yaml.example

A YAML blueprint illustrating the same dual-mode configuration in a structured format.

**Use when:**
- You prefer YAML over dotenv files.
- You're integrating with Kubernetes ConfigMaps or similar orchestration tools.
- You need a human-readable reference for all available settings.

**Note:** The `start.sh` script does not natively parse YAML. This file serves as documentation and a template for converting to environment variables or ConfigMaps.

## Configuration precedence

`start.sh` resolves configuration values in this order (highest to lowest priority):

1. **Command-line arguments** – e.g., `--mode dual`
2. **Environment file** – loaded via `--env-file path/to/.env`
3. **Shell environment variables** – exported before running the script
4. **Default values** – hardcoded in the script

Example showing precedence:

```bash
# .env file contains:
WORKER_MODE=validation
INFERENCE_PORT=9000

# Command-line overrides:
./start.sh --env-file .env --mode dual --inference-port 8080

# Result: mode=dual, port=8080 (CLI wins)
```

## Security best practices

- **Never commit actual secrets** to version control. Always copy `.example` files, fill in your credentials, and add the non-example versions to `.gitignore`.
- Store `HF_TOKEN` and `FLOCK_API_KEY` in secure secret stores (Kubernetes Secrets, AWS Secrets Manager, HashiCorp Vault, etc.).
- When using TLS client certificates for telemetry, mount them read-only into containers.
- Rotate credentials regularly and update the mounted secrets or environment variables.

## Quick start examples

### Validation-only with Docker

```bash
cd config-samples
cp validation.env.example my-validator.env
# Edit my-validator.env
cd ..
docker run --rm --env-file config-samples/my-validator.env \
  flock-validator:gpu
```

### Dual-mode with Kubernetes

```bash
kubectl create configmap flock-config \
  --from-env-file=config-samples/dual-mode.env.example

kubectl create secret generic flock-secrets \
  --from-literal=HF_TOKEN=hf_xxx \
  --from-literal=FLOCK_API_KEY=xxx

# Then reference them in your Deployment spec
```

## Additional resources

- [Main README](../README.md) – comprehensive guide covering all worker modes
- [Operations Guide](../OPERATIONS.md) – production deployment patterns and troubleshooting
- [Changelog](../CHANGELOG.md) – history of features and changes
