# Model Cache Manager Integration Guide

## Overview

The Model Cache Manager (`src/core/model_cache.py`) provides comprehensive caching for Hugging Face models and LoRA adapters with configurable eviction policies and metadata persistence.

## Configuration

### Environment Variables

Configure the cache manager using these environment variables:

```bash
# Enable/disable cache management (default: true)
export CACHE_ENABLED=true

# Cache directory path (default: ~/.cache/huggingface)
export CACHE_DIR=/path/to/cache

# Maximum cache size in GB (default: 100)
export CACHE_MAX_SIZE_GB=50

# Eviction strategy: LRU, FIFO, or SIZE (default: LRU)
export CACHE_EVICTION_STRATEGY=LRU

# Enable/disable automatic eviction on overflow (default: true)
export CACHE_AUTO_EVICT=true

# Hugging Face authentication token
export HF_TOKEN=your_token_here
```

### Eviction Strategies

- **LRU (Least Recently Used)**: Evicts models that haven't been accessed recently
- **FIFO (First In First Out)**: Evicts oldest downloaded models first
- **SIZE**: Evicts largest models first to free up maximum space

## Basic Usage

### Initialize the Cache Manager

```python
from src.core.model_cache import ModelCacheManager

# Using environment variables
manager = ModelCacheManager()

# Or with explicit configuration
manager = ModelCacheManager(
    cache_dir="/custom/cache",
    max_size_gb=75,
    eviction_strategy="LRU",
    hf_token="your_token"
)
```

### Download and Cache Models

```python
# Download a base model
cache_path = manager.download_model(
    model_id="meta-llama/Llama-2-7b-hf",
    model_type="base",
    revision="main"
)

# Download a LoRA adapter
adapter_path = manager.download_model(
    model_id="username/lora-adapter",
    model_type="lora",
    revision="v1.0",
    adapter_base_model="meta-llama/Llama-2-7b-hf"
)
```

### Query Cache State

```python
# Check if a model is cached
if manager.is_model_cached("meta-llama/Llama-2-7b-hf", "main"):
    print("Model is cached!")

# Get model information
info = manager.get_model_info("meta-llama/Llama-2-7b-hf", "main")
print(f"Model size: {info.size_bytes / 1024**3:.2f} GB")
print(f"Last accessed: {info.last_accessed}")

# List all cached models
models = manager.list_models()
for model in models:
    print(f"{model['model_id']}: {model['size_mb']:.2f} MB")

# Get cache statistics
stats = manager.get_cache_stats()
print(f"Total size: {stats['total_size_gb']:.2f} GB")
print(f"Utilization: {stats['utilization_percent']:.1f}%")
print(f"Active models: {stats['active_count']}/{stats['model_count']}")
```

### Mark Models as Active/Inactive

Models marked as active are protected from eviction:

```python
# Mark a model as active when loading
manager.mark_active("meta-llama/Llama-2-7b-hf", "main", active=True)

# Mark as inactive when done
manager.mark_active("meta-llama/Llama-2-7b-hf", "main", active=False)

# Or use convenience methods
success, path = manager.request_load_model(
    "meta-llama/Llama-2-7b-hf",
    "base",
    "main"
)  # Automatically marks as active

manager.request_unload_model("meta-llama/Llama-2-7b-hf", "main")  # Marks inactive
```

### Manual Eviction

```python
# Evict a specific model (fails if active)
success = manager.evict_model("old/model", "main")

# Force eviction even if active
success = manager.evict_model("old/model", "main", force=True)

# Trigger auto-eviction manually
evicted_count = manager.auto_evict()
print(f"Evicted {evicted_count} models")

# Clear entire cache
cleared = manager.clear_cache(force=True)
```

### LoRA Adapter Workflows

```python
# Prepare for LoRA merging (downloads both adapter and base if needed)
success, adapter_path, base_path = manager.prepare_lora_merge(
    adapter_id="username/lora-adapter",
    base_model_id="meta-llama/Llama-2-7b-hf",
    revision="v1.0"
)

if success:
    # Both models are now cached and marked active
    # Proceed with merging...
    pass
```

## Integration with ValidationRunner

The cache manager can be integrated into the validation workflow:

```python
from src.core.model_cache import ModelCacheManager
from src.core.validation_runner import ValidationRunner

# Initialize cache manager
cache_manager = ModelCacheManager()

# Create validation runner
runner = ValidationRunner(hf_token="your_token")

# Before model evaluation
model_id = "meta-llama/Llama-2-7b-hf"
revision = "main"

# Check if model is cached
if cache_manager.is_model_cached(model_id, revision):
    cache_manager.mark_active(model_id, revision)
else:
    # Download and cache will happen via HuggingFace hub
    # Consider pre-downloading via cache manager for better tracking
    pass

# Run evaluation
result = runner.evaluate_model(...)

# After evaluation, mark as inactive
cache_manager.mark_active(model_id, revision, active=False)

# Periodically clean up cache
if cache_manager.get_cache_stats()["utilization_percent"] > 90:
    cache_manager.auto_evict()
```

## Fed Ledger Integration

The cache manager integrates with the Fed Ledger client for coordinated cache management:

```python
from src.client.fed_ledger import FedLedger
from src.core.model_cache import ModelCacheManager

fed_ledger = FedLedger(api_key="your_key")
cache_manager = ModelCacheManager()

# Report cache state to backend
stats = cache_manager.get_cache_stats()
fed_ledger.report_cache_state(stats, worker_id="worker-1")

# Pull and execute cache commands
commands = fed_ledger.get_cache_commands(worker_id="worker-1")
if commands:
    for cmd in commands:
        if cmd["command"] == "download":
            cache_manager.download_model(
                model_id=cmd["model_id"],
                model_type=cmd.get("model_type", "base"),
                revision=cmd.get("revision", "main")
            )
            fed_ledger.execute_cache_command(cmd["id"], "completed")
        
        elif cmd["command"] == "evict":
            success = cache_manager.evict_model(
                model_id=cmd["model_id"],
                revision=cmd.get("revision", "main")
            )
            status = "completed" if success else "failed"
            fed_ledger.execute_cache_command(cmd["id"], status)
        
        elif cmd["command"] == "list":
            models = cache_manager.list_models()
            fed_ledger.execute_cache_command(
                cmd["id"],
                "completed",
                {"models": models}
            )
```

## Thread Safety

The cache manager is thread-safe and can be used in multi-threaded environments:

```python
from threading import Thread

manager = ModelCacheManager()

def worker(model_id):
    success, path = manager.request_load_model(model_id, "base", "main")
    if success:
        # Use model...
        manager.request_unload_model(model_id, "main")

threads = [
    Thread(target=worker, args=(f"model-{i}",))
    for i in range(4)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Best Practices

1. **Mark models active during use**: Always mark models as active when loaded to prevent accidental eviction
2. **Mark inactive when done**: Mark models inactive after use to allow cache cleanup
3. **Monitor cache utilization**: Periodically check cache stats and trigger manual eviction if needed
4. **Configure appropriate limits**: Set `CACHE_MAX_SIZE_GB` based on available disk space
5. **Use LRU for validation workers**: The LRU strategy works well for validation workloads
6. **Pre-download common models**: Use the cache manager to pre-download frequently used base models
7. **Report to Fed Ledger**: Regularly report cache state for coordinated multi-worker management

## Troubleshooting

### Cache not evicting automatically

Check that `CACHE_AUTO_EVICT=true` and verify the max size is being reached:

```python
stats = manager.get_cache_stats()
print(f"Utilization: {stats['utilization_percent']:.1f}%")
print(f"Auto evict enabled: {stats['auto_evict_enabled']}")
```

### Models getting evicted unexpectedly

Ensure models are marked as active during use:

```python
# Before loading
manager.mark_active(model_id, revision, active=True)

# After done
manager.mark_active(model_id, revision, active=False)
```

### Manifest corruption

If the manifest becomes corrupted, it will be automatically reset. You can also manually delete it:

```bash
rm ~/.cache/huggingface/cache_manifest.json
```

### Permission errors

Ensure the cache directory is writable:

```bash
chmod -R u+w ~/.cache/huggingface
```
