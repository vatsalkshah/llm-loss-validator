"""
Model loader for the inference API server.

This module integrates with the existing ModelCacheManager and ValidationRunner
to provide on-demand model loading and management for inference requests.
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
from threading import Lock

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from core.model_cache import ModelCacheManager


class ModelNotCachedError(Exception):
    """Raised when a requested model is not found in the cache."""


class ModelLoader:
    """
    Manages model loading and caching for inference.
    
    This class provides thread-safe model loading with integration to
    the existing cache management system. Models are loaded on-demand
    and kept in memory until explicitly unloaded.
    """
    
    def __init__(
        self,
        cache_manager: ModelCacheManager,
        hf_token: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        reject_non_cached: bool = False,
    ):
        """
        Initialize the ModelLoader.
        
        Args:
            cache_manager: ModelCacheManager instance for cache operations
            hf_token: HuggingFace authentication token
            device: Device to load models on (cpu, cuda, or auto)
            torch_dtype: Torch dtype for model loading (auto, float16, bfloat16, float32)
            reject_non_cached: Reject requests for non-cached models
        """
        self.cache_manager = cache_manager
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.reject_non_cached = reject_non_cached
        
        # In-memory loaded models
        self._loaded_models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
        self._lock = Lock()
        
        logger.info(f"ModelLoader initialized with device={device}, dtype={torch_dtype}")
    
    def _resolve_torch_dtype(self) -> torch.dtype:
        """Resolve the torch dtype based on configuration and device."""
        if self.torch_dtype_str == "auto":
            if self.device == "cpu" or not torch.cuda.is_available():
                return torch.float32
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.torch_dtype_str == "float16":
            return torch.float16
        elif self.torch_dtype_str == "bfloat16":
            return torch.bfloat16
        elif self.torch_dtype_str == "float32":
            return torch.float32
        else:
            return torch.float32
    
    def _get_cache_key(self, model_id: str, revision: str = "main") -> str:
        """Generate a cache key for a model."""
        return f"{model_id}@{revision}"
    
    def is_model_loaded(self, model_id: str, revision: str = "main") -> bool:
        """Check if a model is currently loaded in memory."""
        cache_key = self._get_cache_key(model_id, revision)
        with self._lock:
            return cache_key in self._loaded_models
    
    def get_loaded_model(
        self,
        model_id: str,
        revision: str = "main"
    ) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
        """Retrieve a loaded model from memory."""
        cache_key = self._get_cache_key(model_id, revision)
        with self._lock:
            return self._loaded_models.get(cache_key)
    
    def load_tokenizer(
        self,
        model_name_or_path: str,
        revision: str = "main"
    ) -> AutoTokenizer:
        """
        Load and configure a tokenizer.
        
        Args:
            model_name_or_path: Model identifier or path
            revision: Git revision
        
        Returns:
            Configured AutoTokenizer instance
        """
        tokenizer_kwargs = dict(
            use_fast=True,
        )
        path_obj = Path(model_name_or_path)
        if not path_obj.exists():
            tokenizer_kwargs["revision"] = revision
            if self.hf_token:
                tokenizer_kwargs["token"] = self.hf_token
        else:
            model_name_or_path = str(path_obj)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        
        # Special handling for specific tokenizers
        if "gemma" in model_name_or_path.lower():
            tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
            )
        
        if tokenizer.__class__.__name__ == "QWenTokenizer":
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.eod_id
            tokenizer.eos_token_id = tokenizer.eod_id
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
        assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
        
        return tokenizer
    
    def _resolve_cache_path(self, model_id: str, revision: str = "main") -> Optional[Path]:
        """Resolve the local cache path for a model if available."""
        info = self.cache_manager.get_model_info(model_id, revision)
        if info is None:
            return None
        return Path(info.cache_path)

    def _determine_model_sources(
        self,
        model_id: str,
        cache_path: Path,
        revision: str = "main"
    ) -> Tuple[str, str, bool, Optional[str]]:
        """
        Determine tokenizer and model sources, handling LoRA adapters.
        
        Args:
            model_id: Requested model identifier
            cache_path: Local cache path of the requested model
            revision: Model revision
        
        Returns:
            Tuple of (tokenizer_source, model_source, is_lora, adapter_path)
        """
        adapter_config_path = cache_path / "adapter_config.json"
        if adapter_config_path.exists():
            logger.info(f"Model {model_id} identified as LoRA via adapter_config.json")
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            base_model_id = adapter_config.get("base_model_name_or_path")
            if not base_model_id:
                raise ValueError(
                    f"LoRA model {model_id} does not specify 'base_model_name_or_path'"
                )

            base_cache_path = self._resolve_cache_path(base_model_id, revision)
            if base_cache_path is None:
                # Attempt main revision
                base_cache_path = self._resolve_cache_path(base_model_id, "main")

            tokenizer_source = str(base_cache_path) if base_cache_path else base_model_id
            model_source = tokenizer_source
            adapter_source = str(cache_path)
            return tokenizer_source, model_source, True, adapter_source

        # Fallback to using the cached path directly
        return str(cache_path), str(cache_path), False, None
    
    def load_model(
        self,
        model_id: str,
        revision: str = "main",
        force_reload: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and tokenizer, using cache when possible.
        
        Args:
            model_id: Model identifier
            revision: Git revision
            force_reload: Force reload even if already in memory
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            ValueError: If model is not cached and cache manager rejects non-cached models
            Exception: If model loading fails
        """
        cache_key = self._get_cache_key(model_id, revision)
        
        # Check if already loaded
        if not force_reload and self.is_model_loaded(model_id, revision):
            logger.info(f"Model {cache_key} already loaded, returning from memory")
            return self.get_loaded_model(model_id, revision)
        
        logger.info(f"Loading model: {model_id} (revision: {revision})")
        
        # Check if model is cached
        if not self.cache_manager.is_model_cached(model_id, revision):
            logger.warning(f"Model {model_id} is not in cache")
            if self.reject_non_cached:
                logger.error(f"Model {model_id} requested but not cached")
                raise ModelNotCachedError(f"Model {model_id} not cached")
            # Attempt to download model if allowed
            logger.info(f"Attempting to download model {model_id}")
            try:
                self.cache_manager.download_model(
                    model_id=model_id,
                    model_type="base",
                    revision=revision,
                    token=self.hf_token,
                )
            except Exception as e:
                logger.error(f"Failed to download model {model_id}: {e}")
                raise
        
        # Resolve cached model path
        cache_path = self._resolve_cache_path(model_id, revision)
        if cache_path is None:
            raise ValueError(f"Model {model_id} cache path could not be resolved")
        
        # Determine tokenizer and model sources
        tokenizer_source, model_source, is_lora, adapter_path = self._determine_model_sources(
            model_id, cache_path, revision
        )
        
        # Load tokenizer from resolved source
        tokenizer = self.load_tokenizer(tokenizer_source)
        
        # Load model
        torch_dtype = self._resolve_torch_dtype()
        
        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_cache=True,  # Enable KV cache for inference
            device_map=self.device if self.device != "auto" else "auto",
        )
        
        model_path_obj = Path(model_source)
        if not model_path_obj.exists():
            if self.hf_token:
                model_kwargs["token"] = self.hf_token
            model_kwargs["revision"] = revision
        else:
            model_source = str(model_path_obj)
        
        if is_lora:
            logger.info(f"Loading base model from {model_source} and LoRA adapter from {adapter_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs
            )
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map=model_kwargs["device_map"],
            )
            model = model.merge_and_unload()
            logger.info("Loaded model with LoRA adapter merged")
        else:
            logger.info(f"Loading full model from {model_source}")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs
            )
        
        # Enable eval mode
        model.eval()
        
        # Store in memory
        with self._lock:
            self._loaded_models[cache_key] = (model, tokenizer)
        
        # Mark as active in cache
        self.cache_manager.mark_active(model_id, revision, active=True)
        
        logger.info(f"Successfully loaded model {cache_key}")
        return model, tokenizer
    
    def unload_model(self, model_id: str, revision: str = "main") -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model identifier
            revision: Git revision
        
        Returns:
            True if model was unloaded, False if not found
        """
        cache_key = self._get_cache_key(model_id, revision)
        
        with self._lock:
            if cache_key not in self._loaded_models:
                logger.warning(f"Model {cache_key} not loaded")
                return False
            
            model, tokenizer = self._loaded_models[cache_key]
            
            # Move to CPU and delete
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            del tokenizer
            
            # Remove from loaded models
            del self._loaded_models[cache_key]
        
        # Mark as inactive in cache
        self.cache_manager.mark_active(model_id, revision, active=False)
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model {cache_key}")
        return True
    
    def list_loaded_models(self) -> list:
        """List all currently loaded models."""
        with self._lock:
            return list(self._loaded_models.keys())
