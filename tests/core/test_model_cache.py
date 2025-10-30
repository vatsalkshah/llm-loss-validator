import json
from pathlib import Path

import pytest

from src.core.model_cache import ModelCacheManager


def create_dummy_dir(tmp_path: Path, name: str, size: int) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "weights.bin").write_bytes(b"0" * size)
    return path


def test_register_and_stats(tmp_path: Path):
    cache = ModelCacheManager(tmp_path, limit_bytes=10_000)
    model_dir = create_dummy_dir(tmp_path, "model-a", 1024)
    cache.register("model-a", model_dir)

    stats = cache.stats()
    assert stats["artefact_count"] == 1
    assert stats["size_bytes"] >= 1024


def test_prune_removes_oldest(tmp_path: Path):
    cache = ModelCacheManager(tmp_path, limit_bytes=1500)
    dir_a = create_dummy_dir(tmp_path, "model-a", 800)
    dir_b = create_dummy_dir(tmp_path, "model-b", 800)
    cache.register("model-a", dir_a)
    cache.register("model-b", dir_b)

    cache.prune()
    entries = cache.list_entries()
    assert len(entries) == 1
    remaining = next(iter(entries))
    assert remaining in {"model-a", "model-b"}


def test_index_persistence(tmp_path: Path):
    cache = ModelCacheManager(tmp_path, limit_bytes=None)
    model_dir = create_dummy_dir(tmp_path, "model-a", 100)
    cache.register("model-a", model_dir)

    index_path = cache.cache_dir / cache.INDEX_FILE
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert "model-a" in data

    # rebuild manager and ensure entry restored
    cache2 = ModelCacheManager(tmp_path, limit_bytes=None)
    assert "model-a" in cache2.list_entries()


def test_reserve_space(tmp_path: Path):
    cache = ModelCacheManager(tmp_path, limit_bytes=1024)
    with cache.reserve_space(500):
        pass
    # no exception should be raised even when cache empty


if __name__ == "__main__":
    pytest.main([__file__])
