"""Model cache utilities shared between validation and inference workers."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional


@dataclass(slots=True)
class CacheEntry:
    """Metadata describing a single cached artefact."""

    model_id: str
    path: Path
    size_bytes: int
    last_access: float

    def touch(self) -> None:
        self.last_access = time.time()


class CachePruneError(RuntimeError):
    """Raised when the cache manager fails to delete a directory."""


class ModelCacheManager:
    """Track usage of the Hugging Face cache and enforce disk limits."""

    INDEX_FILE = ".cache-index.json"

    def __init__(self, cache_dir: str | Path, limit_bytes: Optional[int] = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limit_bytes = limit_bytes
        self._lock = threading.RLock()
        self._index_path = self.cache_dir / self.INDEX_FILE
        self._entries: Dict[str, CacheEntry] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text())
        except json.JSONDecodeError:
            # Corrupt index; start over.
            return

        for model_id, payload in data.items():
            path = Path(payload["path"])
            if path.exists():
                entry = CacheEntry(
                    model_id=model_id,
                    path=path,
                    size_bytes=int(payload.get("size_bytes", 0)),
                    last_access=float(payload.get("last_access", time.time())),
                )
                self._entries[model_id] = entry

    def _save_index(self) -> None:
        data = {
            model_id: {
                "path": str(entry.path),
                "size_bytes": entry.size_bytes,
                "last_access": entry.last_access,
            }
            for model_id, entry in self._entries.items()
        }
        self._index_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def _compute_dir_size(self, path: Path) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                try:
                    total += (Path(root) / filename).stat().st_size
                except FileNotFoundError:
                    continue
        return total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(self, model_id: str, path: str | Path) -> CacheEntry:
        """Register a cached artefact and update accounting information."""

        with self._lock:
            resolved = Path(path).resolve()
            size = self._compute_dir_size(resolved)
            entry = CacheEntry(model_id=model_id, path=resolved, size_bytes=size, last_access=time.time())
            self._entries[model_id] = entry
            self._save_index()
            if self.limit_bytes is not None:
                self.prune()
            return entry

    def touch(self, model_id: str) -> None:
        """Update the last access time for an artefact."""

        with self._lock:
            if model_id in self._entries:
                self._entries[model_id].touch()
                self._save_index()

    def evict(self, model_id: str) -> None:
        """Remove an artefact from disk and the index."""

        with self._lock:
            entry = self._entries.pop(model_id, None)
            if not entry:
                return
            try:
                shutil.rmtree(entry.path, ignore_errors=True)
            finally:
                self._save_index()

    def prune(self) -> None:
        """Ensure the cache stays within the configured disk limit."""

        if self.limit_bytes is None:
            return

        with self._lock:
            total = sum(entry.size_bytes for entry in self._entries.values())
            if total <= self.limit_bytes:
                return

            for model_id, entry in sorted(self._entries.items(), key=lambda item: item[1].last_access):
                try:
                    shutil.rmtree(entry.path, ignore_errors=True)
                except OSError as exc:  # pragma: no cover - platform dependent
                    raise CachePruneError(str(exc)) from exc
                total -= entry.size_bytes
                del self._entries[model_id]
                if total <= self.limit_bytes:
                    break
            self._save_index()

    @contextmanager
    def reserve_space(self, required_bytes: int) -> Iterator[None]:
        """Context manager that prunes proactively before downloading.

        This is useful when a caller knows how large the download will be.
        """

        with self._lock:
            if self.limit_bytes is not None:
                while self.free_space() < required_bytes:
                    self.prune()
        yield

    # ------------------------------------------------------------------
    # Telemetry / diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int | float]:
        with self._lock:
            total_bytes = sum(entry.size_bytes for entry in self._entries.values())
            headroom = None if self.limit_bytes is None else max(self.limit_bytes - total_bytes, 0)
            return {
                "artefact_count": len(self._entries),
                "size_bytes": total_bytes,
                "limit_bytes": self.limit_bytes if self.limit_bytes is not None else 0,
                "headroom_bytes": headroom if headroom is not None else 0,
            }

    def free_space(self) -> int:
        if self.limit_bytes is None:
            return int(1e18)
        return max(self.limit_bytes - self.stats()["size_bytes"], 0)

    def list_entries(self) -> Dict[str, CacheEntry]:
        with self._lock:
            return dict(self._entries)


__all__ = [
    "CacheEntry",
    "CachePruneError",
    "ModelCacheManager",
]
