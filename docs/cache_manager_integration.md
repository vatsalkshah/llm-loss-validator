# Model Cache Integration

The `ModelCacheManager` module centralises the logic for provisioning, tracking, and pruning artefacts downloaded from the Hugging Face Hub. It is designed to be lightweight, thread-safe, and friendly to both validation runners and inference services operating inside the same container.

## Responsibilities

1. **Index management** – persist a JSON index (`.cache-index.json`) that records path, size, and last access time for each artefact.
2. **Usage accounting** – compute aggregate cache size and file counts on demand without repeatedly traversing the filesystem.
3. **Limit enforcement** – discard least-recently-used artefacts when `CACHE_LIMIT_BYTES` (or the convenience `CACHE_LIMIT_GB`) threshold is exceeded.
4. **Observability hooks** – expose `stats()` for telemetry reporters to include cache metadata in heartbeat payloads.
5. **Concurrency safety** – guard mutating operations with a re-entrant lock so validation and inference threads can share the manager.

## Integration points

### Validation Runner

```python
cache = ModelCacheManager(cache_dir, limit_bytes)
cache.register("meta-llama/Llama-2-7b", local_path)
with cache.reserve_space(required_bytes):
    download_weights()
result = runner.evaluate(assignment)
cache.touch("meta-llama/Llama-2-7b")
```

The validation runner updates the cache after successfully materialising an artefact. When the assignment is complete the runner calls `cache.touch()` so that dual-mode inference knows the checkpoint is still hot.

### Inference API

```python
model = loader.get_or_load(model_id)
cache.touch(model_id)
response = pipeline.generate(request)
```

The inference service piggybacks on the same cache manager to avoid duplicate downloads. If the cache limit is reached the manager evicts the coldest artefacts before returning a handle.

### Telemetry

```python
telemetry.emit(
    "cache",
    cache.stats()
)
```

Telemetry clients serialise the `stats()` output (bytes used, number of artefacts, headroom) to give operators visibility into disk pressure.

## Limit enforcement algorithm

1. Recompute current usage (`size_bytes`).
2. If below the limit, stop.
3. Sort indexed artefacts by `last_access` ascending.
4. Remove directories until the limit is satisfied.
5. Persist the updated index.

This approach keeps the implementation deterministic and very easy to test. The manager intentionally avoids inotify or platform-specific features to remain portable between Linux, macOS, and Windows runners.

## Error handling

- Missing directories are created automatically on initialisation.
- IOErrors during pruning raise `CachePruneError` (a `RuntimeError` subclass) so callers can surface actionable diagnostics.
- The manager logs size calculations at `DEBUG` level, but the logging hook is optional to keep dependencies minimal.
