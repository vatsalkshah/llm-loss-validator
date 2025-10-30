"""Utilities for loading models used by the inference API."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from ..core.model_cache import ModelCacheManager


class DummyModel:
    """Lightweight fallback model used in tests and development."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return f"{prompt} -> response from {self.model_id}"[: max(1, max_tokens)]


FactoryFn = Callable[[str], DummyModel]


class ModelLoader:
    """Thread-safe cache of model instances."""

    def __init__(
        self,
        cache: Optional[ModelCacheManager] = None,
        factory: Optional[FactoryFn] = None,
    ) -> None:
        self._cache = cache
        self._factory = factory or (lambda model_id: DummyModel(model_id))
        self._instances: Dict[str, DummyModel] = {}
        self._lock = threading.RLock()

    def get(self, model_id: str) -> DummyModel:
        with self._lock:
            if model_id in self._instances:
                model = self._instances[model_id]
                if self._cache is not None:
                    self._cache.touch(model_id)
                return model

            model = self._factory(model_id)
            self._instances[model_id] = model
            if self._cache is not None:
                self._cache.register(model_id, self._cache.cache_dir)
                self._cache.touch(model_id)
            return model

    def unload(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._instances:
                del self._instances[model_id]
                if self._cache is not None:
                    self._cache.evict(model_id)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "loaded_models": len(self._instances),
                "last_load_time": time.time(),
            }


__all__ = ["ModelLoader", "DummyModel"]
