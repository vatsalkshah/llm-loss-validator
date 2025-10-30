"""
Model Cache Manager Module.

This module provides comprehensive cache management for Hugging Face models and adapters,
including download tracking, eviction policies, and metadata persistence. It integrates
with the validation worker to optimize model loading and manage disk usage.

Key Features:
    - Track cached models and LoRA adapters with detailed metadata
    - Configurable cache directory and size limits via environment variables
    - Multiple eviction strategies (LRU, FIFO, size-based)
    - JSON-based manifest for persistence across restarts
    - APIs for querying, loading, and unloading models
    - Hooks for LoRA adapter merging workflows
    - Thread-safe operations for concurrent access

Environment Variables:
    CACHE_DIR: Directory path for model cache (default: ~/.cache/huggingface)
    CACHE_MAX_SIZE_GB: Maximum cache size in GB (default: 100)
    CACHE_EVICTION_STRATEGY: Eviction strategy - LRU, FIFO, SIZE (default: LRU)
    CACHE_ENABLED: Enable/disable cache management (default: true)
    CACHE_AUTO_EVICT: Enable/disable automatic eviction on overflow (default: true)
    HF_TOKEN: Hugging Face authentication token for model downloads
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from threading import Lock

from loguru import logger
from huggingface_hub import snapshot_download, hf_hub_download


@dataclass
class CachedModelInfo:
    """Metadata for a cached model or adapter."""
    model_id: str
    model_type: str  # 'base', 'lora', 'full'
    size_bytes: int
    download_timestamp: float
    last_accessed: float
    revision: str
    cache_path: str
    is_active: bool = False
    adapter_base_model: Optional[str] = None  # For LoRA adapters
    download_duration: Optional[float] = None


class EvictionStrategy:
    """Supported cache eviction strategies."""
    LRU = "LRU"  # Least Recently Used
    FIFO = "FIFO"  # First In First Out
    SIZE = "SIZE"  # Largest models first


class ModelCacheManager:
    """
    Manages caching of Hugging Face models and adapters.
    
    This class encapsulates all cache operations including download tracking,
    eviction based on configurable policies, and metadata persistence. It provides
    thread-safe operations and integrates seamlessly with the validation workflow.
    
    The cache manager maintains a JSON manifest that persists across restarts,
    tracking each model's size, access patterns, and relationship to other models
    (e.g., LoRA adapters and their base models).
    
    Usage:
        manager = ModelCacheManager()
        
        # Download and track a model
        cache_path = manager.download_model("meta-llama/Llama-2-7b", "base")
        
        # Mark model as actively in use
        manager.mark_active("meta-llama/Llama-2-7b")
        
        # List all cached models
        models = manager.list_models()
        
        # Perform cleanup if needed
        manager.auto_evict()
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size_gb: Optional[float] = None,
        eviction_strategy: Optional[str] = None,
        manifest_file: Optional[str] = None,
        enabled: Optional[bool] = None,
        hf_token: Optional[str] = None,
        auto_evict_enabled: Optional[bool] = None,
    ):
        """
        Initialize the ModelCacheManager.
        
        Args:
            cache_dir: Directory for model cache (default from CACHE_DIR env var)
            max_size_gb: Maximum cache size in GB (default from CACHE_MAX_SIZE_GB)
            eviction_strategy: Eviction strategy to use (default from CACHE_EVICTION_STRATEGY)
            manifest_file: Path to manifest file (default: <cache_dir>/cache_manifest.json)
            enabled: Override for cache enablement (default from CACHE_ENABLED)
            hf_token: Hugging Face token to use for downloads (default from HF_TOKEN)
            auto_evict_enabled: Enable automatic eviction on cache overflow (default True)
        """
        env_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.enabled = env_enabled if enabled is None else enabled
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        env_auto_evict = os.getenv("CACHE_AUTO_EVICT", "true").lower() == "true"
        self.auto_evict_enabled = env_auto_evict if auto_evict_enabled is None else auto_evict_enabled
        
        # Configuration
        default_cache = os.path.expanduser("~/.cache/huggingface")
        self.cache_dir = Path(cache_dir or os.getenv("CACHE_DIR", default_cache))
        self.max_size_bytes = int(
            float(max_size_gb or os.getenv("CACHE_MAX_SIZE_GB", "100")) * 1024 ** 3
        )
        self.eviction_strategy = (
            eviction_strategy or os.getenv("CACHE_EVICTION_STRATEGY", EvictionStrategy.LRU)
        )
        
        # Manifest file
        self.manifest_path = Path(
            manifest_file or self.cache_dir / "cache_manifest.json"
        )
        
        # Thread safety
        self._lock = Lock()
        
        # Cache state
        self.manifest: Dict[str, CachedModelInfo] = {}
        
        # Initialize
        self._ensure_cache_dir()
        self._load_manifest()
        
        logger.info(
            f"ModelCacheManager initialized: enabled={self.enabled}, "
            f"cache_dir={self.cache_dir}, max_size={self.max_size_bytes / 1024**3:.2f}GB, "
            f"strategy={self.eviction_strategy}, auto_evict={self.auto_evict_enabled}"
        )
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_manifest(self) -> None:
        """Load cache manifest from disk."""
        if not self.manifest_path.exists():
            logger.info("No existing manifest found, starting with empty cache")
            self.manifest = {}
            return
        
        try:
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
            
            self.manifest = {
                model_id: CachedModelInfo(**info)
                for model_id, info in data.items()
            }
            logger.info(f"Loaded manifest with {len(self.manifest)} entries")
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}, starting fresh")
            self.manifest = {}
    
    def _save_manifest(self) -> None:
        """Persist cache manifest to disk."""
        try:
            data = {
                model_id: asdict(info)
                for model_id, info in self.manifest.items()
            }
            
            # Atomic write
            temp_path = self.manifest_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.manifest_path)
            
            logger.debug(f"Saved manifest with {len(self.manifest)} entries")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def _resolve_token(self, token: Optional[str]) -> Optional[str]:
        """Resolve the Hugging Face token to use for download operations."""
        return token or self.hf_token
    
    def get_cache_size(self) -> int:
        """
        Calculate current total cache size in bytes.
        
        Returns:
            Total size of all cached models in bytes
        """
        with self._lock:
            return sum(info.size_bytes for info in self.manifest.values())
    
    def get_cache_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics including size, count, and utilization
        """
        with self._lock:
            total_size = self.get_cache_size()
            return {
                "total_size_bytes": total_size,
                "total_size_gb": total_size / 1024 ** 3,
                "max_size_gb": self.max_size_bytes / 1024 ** 3,
                "utilization_percent": (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0,
                "model_count": len(self.manifest),
                "active_count": sum(1 for info in self.manifest.values() if info.is_active),
                "eviction_strategy": self.eviction_strategy,
                "auto_evict_enabled": self.auto_evict_enabled,
                "enabled": self.enabled,
            }
    
    def list_models(self, include_inactive: bool = True) -> List[Dict]:
        """
        List all cached models with their metadata.
        
        Args:
            include_inactive: Whether to include inactive models
        
        Returns:
            List of model information dictionaries
        """
        with self._lock:
            models = []
            for model_id, info in self.manifest.items():
                if not include_inactive and not info.is_active:
                    continue
                
                model_dict = asdict(info)
                model_dict["size_mb"] = info.size_bytes / 1024 ** 2
                model_dict["last_accessed_str"] = datetime.fromtimestamp(
                    info.last_accessed
                ).isoformat()
                models.append(model_dict)
            
            return models
    
    def get_model_info(self, model_id: str, revision: str = "main") -> Optional[CachedModelInfo]:
        """
        Get cached model information.
        
        Args:
            model_id: Hugging Face model ID
            revision: Model revision/branch
        
        Returns:
            CachedModelInfo if found, None otherwise
        """
        cache_key = self._make_cache_key(model_id, revision)
        with self._lock:
            return self.manifest.get(cache_key)
    
    def is_model_cached(self, model_id: str, revision: str = "main") -> bool:
        """
        Check if a model is already cached.
        
        Args:
            model_id: Hugging Face model ID
            revision: Model revision/branch
        
        Returns:
            True if model is cached, False otherwise
        """
        return self.get_model_info(model_id, revision) is not None
    
    def mark_active(
        self,
        model_id: str,
        revision: str = "main",
        active: bool = True
    ) -> bool:
        """
        Mark a model as actively in use (or inactive).
        
        This updates the last_accessed timestamp and prevents eviction
        of active models. Should be called when a model is loaded.
        
        Args:
            model_id: Hugging Face model ID
            revision: Model revision/branch
            active: Whether to mark as active or inactive
        
        Returns:
            True if model was found and updated, False otherwise
        """
        cache_key = self._make_cache_key(model_id, revision)
        
        with self._lock:
            if cache_key not in self.manifest:
                logger.warning(f"Cannot mark unknown model as active: {cache_key}")
                return False
            
            self.manifest[cache_key].is_active = active
            self.manifest[cache_key].last_accessed = time.time()
            self._save_manifest()
            
            logger.debug(f"Marked model {'active' if active else 'inactive'}: {cache_key}")
            return True
    
    def download_model(
        self,
        model_id: str,
        model_type: str,
        revision: str = "main",
        token: Optional[str] = None,
        adapter_base_model: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """
        Download a model or adapter and track it in the cache.
        
        This method downloads the model using huggingface_hub's snapshot_download,
        calculates its size, and adds it to the cache manifest. If the cache
        exceeds limits, auto-eviction is triggered.
        
        Args:
            model_id: Hugging Face model ID
            model_type: Type of model - 'base', 'lora', or 'full'
            revision: Model revision/branch
            token: HuggingFace authentication token
            adapter_base_model: Base model ID if this is a LoRA adapter
            force: Force re-download even if cached
        
        Returns:
            Path to the cached model
        
        Raises:
            Exception: If download fails or cache operations fail
        """
        if not self.enabled:
            logger.info("Cache manager disabled, downloading without tracking")
            return snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=self._resolve_token(token),
            )
        
        cache_key = self._make_cache_key(model_id, revision)
        
        # Check if already cached
        if not force and self.is_model_cached(model_id, revision):
            logger.info(f"Model already cached: {cache_key}")
            info = self.get_model_info(model_id, revision)
            self.mark_active(model_id, revision)
            return info.cache_path
        
        logger.info(f"Downloading model: {model_id} (revision: {revision}, type: {model_type})")
        
        start_time = time.time()
        
        try:
            # Download using HuggingFace hub
            resolved_token = self._resolve_token(token)
            cache_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=resolved_token,
            )
            
            # Calculate size
            size_bytes = self._calculate_dir_size(cache_path)
            download_duration = time.time() - start_time
            current_time = time.time()
            
            # Create cache entry
            cache_info = CachedModelInfo(
                model_id=model_id,
                model_type=model_type,
                size_bytes=size_bytes,
                download_timestamp=start_time,
                last_accessed=current_time,
                revision=revision,
                cache_path=cache_path,
                is_active=True,
                adapter_base_model=adapter_base_model,
                download_duration=download_duration,
            )
            
            # Add to manifest
            with self._lock:
                self.manifest[cache_key] = cache_info
                self._save_manifest()
            
            logger.info(
                f"Downloaded and cached {cache_key}: "
                f"{size_bytes / 1024**2:.2f} MB in {download_duration:.2f}s"
            )
            
            # Check if we need to evict
            current_size = self.get_cache_size()
            if current_size > self.max_size_bytes:
                if self.auto_evict_enabled:
                    logger.warning("Cache size exceeded, triggering auto-eviction")
                    self.auto_evict()
                else:
                    logger.warning(
                        "Cache size exceeds configured limit but auto eviction is disabled"
                    )
            
            return cache_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise
    
    def download_file(
        self,
        model_id: str,
        filename: str,
        revision: str = "main",
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> str:
        """
        Download a single file from a model repository.
        
        Args:
            model_id: Hugging Face model ID
            filename: Name of the file to download
            revision: Model revision/branch
            token: HuggingFace authentication token
            subfolder: Optional subfolder path
        
        Returns:
            Path to the downloaded file
        """
        logger.info(f"Downloading file {filename} from {model_id}")
        
        resolved_token = self._resolve_token(token)
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            token=resolved_token,
            subfolder=subfolder,
        )
    
    def evict_model(
        self,
        model_id: str,
        revision: str = "main",
        force: bool = False
    ) -> bool:
        """
        Evict a specific model from the cache.
        
        This removes the model from disk and updates the manifest. Active models
        are protected from eviction unless force=True.
        
        Args:
            model_id: Hugging Face model ID
            revision: Model revision/branch
            force: Force eviction even if model is active
        
        Returns:
            True if model was evicted, False otherwise
        """
        cache_key = self._make_cache_key(model_id, revision)
        
        with self._lock:
            if cache_key not in self.manifest:
                logger.warning(f"Cannot evict unknown model: {cache_key}")
                return False
            
            info = self.manifest[cache_key]
            
            if info.is_active and not force:
                logger.warning(f"Cannot evict active model: {cache_key}")
                return False
            
            # Remove from disk
            cache_path = Path(info.cache_path)
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                    logger.info(f"Removed model from disk: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to remove model directory: {e}")
                    return False
            
            # Remove from manifest
            del self.manifest[cache_key]
            self._save_manifest()
            
            logger.info(f"Evicted model: {cache_key} ({info.size_bytes / 1024**2:.2f} MB)")
            return True
    
    def auto_evict(self, target_size: Optional[int] = None, force: bool = False) -> int:
        """
        Automatically evict models based on the configured strategy.
        
        This method evicts models until the cache size is below the target
        (or max_size_bytes if target not specified). Active models are never evicted.
        
        Args:
            target_size: Target size in bytes (default: 90% of max_size_bytes)
            force: Force eviction even if auto eviction is disabled by configuration
        
        Returns:
            Number of models evicted
        """
        if not self.enabled:
            return 0
        
        if not force and not self.auto_evict_enabled:
            logger.debug("Auto eviction disabled by configuration")
            return 0
        
        if target_size is None:
            target_size = int(self.max_size_bytes * 0.9)
        
        current_size = self.get_cache_size()
        
        if current_size <= target_size:
            logger.debug("Cache size within limits, no eviction needed")
            return 0
        
        logger.info(
            f"Starting auto-eviction: current={current_size / 1024**3:.2f}GB, "
            f"target={target_size / 1024**3:.2f}GB, strategy={self.eviction_strategy}"
        )
        
        # Get eviction candidates (non-active models)
        with self._lock:
            candidates = [
                (key, info) for key, info in self.manifest.items()
                if not info.is_active
            ]
        
        if not candidates:
            logger.warning("No eviction candidates available (all models active)")
            return 0
        
        # Sort by strategy
        if self.eviction_strategy == EvictionStrategy.LRU:
            candidates.sort(key=lambda x: x[1].last_accessed)
        elif self.eviction_strategy == EvictionStrategy.FIFO:
            candidates.sort(key=lambda x: x[1].download_timestamp)
        elif self.eviction_strategy == EvictionStrategy.SIZE:
            candidates.sort(key=lambda x: x[1].size_bytes, reverse=True)
        else:
            logger.warning(f"Unknown eviction strategy: {self.eviction_strategy}, using LRU")
            candidates.sort(key=lambda x: x[1].last_accessed)
        
        # Evict until target reached
        evicted_count = 0
        for cache_key, info in candidates:
            if self.get_cache_size() <= target_size:
                break
            
            # Parse model_id and revision from cache_key
            model_id, revision = self._parse_cache_key(cache_key)
            if self.evict_model(model_id, revision):
                evicted_count += 1
        
        logger.info(
            f"Auto-eviction complete: evicted {evicted_count} models, "
            f"new size={self.get_cache_size() / 1024**3:.2f}GB"
        )
        
        return evicted_count
    
    def clear_cache(self, force: bool = False) -> int:
        """
        Clear all cached models.
        
        Args:
            force: If True, clear active models too
        
        Returns:
            Number of models cleared
        """
        logger.warning(f"Clearing cache (force={force})")
        
        with self._lock:
            keys_to_clear = list(self.manifest.keys())
        
        cleared_count = 0
        for cache_key in keys_to_clear:
            model_id, revision = self._parse_cache_key(cache_key)
            if self.evict_model(model_id, revision, force=force):
                cleared_count += 1
        
        logger.info(f"Cache cleared: {cleared_count} models removed")
        return cleared_count
    
    def request_load_model(
        self,
        model_id: str,
        model_type: str,
        revision: str = "main",
    ) -> Tuple[bool, Optional[str]]:
        """
        Request on-demand loading of a model.
        
        This is a high-level API that checks if a model is cached, downloads it
        if necessary, and marks it as active.
        
        Args:
            model_id: Hugging Face model ID
            model_type: Type of model - 'base', 'lora', or 'full'
            revision: Model revision/branch
        
        Returns:
            Tuple of (success, cache_path or error_message)
        """
        try:
            if self.is_model_cached(model_id, revision):
                info = self.get_model_info(model_id, revision)
                self.mark_active(model_id, revision)
                return True, info.cache_path
            
            # Need to download
            cache_path = self.download_model(
                model_id=model_id,
                model_type=model_type,
                revision=revision,
                token=self.hf_token,
            )
            return True, cache_path
            
        except Exception as e:
            error_msg = f"Failed to load model {model_id}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def request_unload_model(
        self,
        model_id: str,
        revision: str = "main"
    ) -> bool:
        """
        Request unloading of a model (mark as inactive).
        
        This doesn't evict the model, just marks it as no longer in use,
        making it eligible for eviction.
        
        Args:
            model_id: Hugging Face model ID
            revision: Model revision/branch
        
        Returns:
            True if model was marked inactive, False otherwise
        """
        return self.mark_active(model_id, revision, active=False)
    
    def prepare_lora_merge(
        self,
        adapter_id: str,
        base_model_id: str,
        revision: str = "main",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Prepare for LoRA adapter merging by ensuring both adapter and base model are cached.
        
        This is a convenience method that downloads both the adapter and its base model
        if needed, marking both as active.
        
        Args:
            adapter_id: Hugging Face adapter repository ID
            base_model_id: Base model ID
            revision: Adapter revision/branch
        
        Returns:
            Tuple of (success, adapter_path, base_model_path)
        """
        logger.info(f"Preparing LoRA merge: adapter={adapter_id}, base={base_model_id}")
        
        try:
            # Load base model
            base_success, base_path = self.request_load_model(
                model_id=base_model_id,
                model_type="base",
                revision="main",
            )
            
            if not base_success:
                return False, None, None
            
            # Load adapter
            adapter_success, adapter_path = self.request_load_model(
                model_id=adapter_id,
                model_type="lora",
                revision=revision,
            )
            
            if not adapter_success:
                return False, None, None
            
            logger.info(f"LoRA merge preparation complete")
            return True, adapter_path, base_path
            
        except Exception as e:
            logger.error(f"Failed to prepare LoRA merge: {e}")
            return False, None, None
    
    def _make_cache_key(self, model_id: str, revision: str) -> str:
        """Create a unique cache key for a model."""
        return f"{model_id}@{revision}"
    
    def _parse_cache_key(self, cache_key: str) -> Tuple[str, str]:
        """Parse a cache key into model_id and revision."""
        if "@" in cache_key:
            model_id, revision = cache_key.rsplit("@", 1)
        else:
            model_id = cache_key
            revision = "main"
        return model_id, revision
    
    def _calculate_dir_size(self, path: str) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        try:
            for entry in Path(path).rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total
