from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import get_config
from src.core.hf_utils import download_lora_config, download_lora_repo


@dataclass
class ModelEntry:
    model_id: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


class ModelRegistry:
    def __init__(self, max_size: Optional[int] = None) -> None:
        cfg = get_config()
        self.max_size = max_size or cfg.model_cache_max
        self.hf_token = cfg.hf_token
        self._cache: "OrderedDict[str, ModelEntry]" = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_size:
            _, entry = self._cache.popitem(last=False)
            try:
                entry.model.cpu()
                del entry.model
            except Exception:
                pass
            logger.info(f"Evicted model from cache: {entry.model_id}")

    def _touch(self, key: str) -> None:
        # move to end (most recently used)
        try:
            self._cache.move_to_end(key)
        except KeyError:
            pass

    def list_loaded(self) -> list[str]:
        return list(self._cache.keys())

    def get(self, model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if model_id in self._cache:
            self._touch(model_id)
            entry = self._cache[model_id]
            return entry.model, entry.tokenizer

        model, tokenizer = self._load(model_id)
        self._cache[model_id] = ModelEntry(model_id=model_id, model=model, tokenizer=tokenizer)
        self._touch(model_id)
        self._evict_if_needed()
        return model, tokenizer

    def _load(self, model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        logger.info(f"Loading model: {model_id}")
        trust_remote_code = True
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs = dict(trust_remote_code=trust_remote_code, torch_dtype=dtype)

        # Detect LoRA adapter
        is_lora = False
        try:
            is_lora = download_lora_config(model_id, revision="main")
        except Exception:
            # treat as full model
            is_lora = False

        if is_lora:
            adapter_config_path = Path("lora/adapter_config.json")
            if not adapter_config_path.exists():
                logger.warning("LoRA adapter_config.json not found after download; loading as base model")
                is_lora = False

        if is_lora:
            with open("lora/adapter_config.json", "r") as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path")
            if not base_model:
                logger.warning("LoRA config missing base_model_name_or_path; loading as full model")
                is_lora = False
            else:
                base = AutoModelForCausalLM.from_pretrained(base_model, token=self.hf_token, **model_kwargs)
                download_lora_repo(model_id, revision="main")
                peft_model = PeftModel.from_pretrained(base, "lora", device_map=None)
                model = peft_model.merge_and_unload()
                tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return model, tokenizer

        # Full model path
        model = AutoModelForCausalLM.from_pretrained(model_id, token=self.hf_token, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


# Singleton
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
