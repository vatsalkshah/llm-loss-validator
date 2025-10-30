"""
Validation Runner Service Module.

This module provides a reusable service layer for model validation,
extracting the core evaluation logic from the CLI to enable
programmatic access by schedulers, inference services, and tests.
"""

import gc
import json
import math
import numbers
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel

from core.collator import SFTDataCollator
from core.dataset import UnifiedSFTDataset
from core.template import template_dict
from core.hf_utils import download_lora_config, download_lora_repo
from core.constant import SUPPORTED_BASE_MODELS
from core.loss import (
    calculate_bpc_bppl_metrics,
    get_token_byte_ratio,
    calculate_bytes_and_tokens,
)
from core.validation_result import ValidationResult
from core.validation_exceptions import (
    InvalidModelException,
    InvalidDatasetException,
    InvalidLoraConfigException,
    ModelParamsExceededException,
)


LOSS_FOR_MODEL_PARAMS_EXCEED = 999.0


class ValidationRunner:
    """
    Service class for executing model validation workflows.

    This class encapsulates the logic for loading models/tokenizers,
    preparing datasets, and running Trainer.evaluate, returning
    structured ValidationResult objects.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the ValidationRunner.

        Args:
            hf_token: HuggingFace authentication token for model access
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if self.hf_token is None:
            raise ValueError(
                "HF_TOKEN is required to download models from HuggingFace"
            )

    def load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        """
        Load and configure a tokenizer from HuggingFace.

        Args:
            model_name_or_path: Model identifier or path

        Returns:
            Configured AutoTokenizer instance
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
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
        logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")
        return tokenizer

    def load_model(
        self,
        model_name_or_path: str,
        lora_only: bool,
        revision: str,
        val_args: TrainingArguments,
        cached_lora: bool,
    ) -> AutoModelForCausalLM:
        """
        Load a model, optionally merging LoRA adapters.

        Args:
            model_name_or_path: Model identifier or path
            lora_only: If True, reject non-LoRA models
            revision: Git revision to load
            val_args: Training arguments (contains device/dtype config)
            cached_lora: Whether LoRA config has been cached locally

        Returns:
            Loaded model instance

        Raises:
            InvalidModelException: If model cannot be loaded or lora_only mismatch
        """
        logger.info(f"Loading model from base model: {model_name_or_path}")

        if val_args.use_cpu:
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if val_args.fp16 else torch.bfloat16

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_cache=False,
            device_map=None,
        )

        if cached_lora:
            logger.info("Repo is a lora weight, loading model with adapter weights")
            with open("lora/adapter_config.json", "r") as f:
                adapter_config = json.load(f)
            base_model = adapter_config["base_model_name_or_path"]
            model = AutoModelForCausalLM.from_pretrained(
                base_model, token=self.hf_token, **model_kwargs
            )
            download_lora_repo(model_name_or_path, revision)
            model = PeftModel.from_pretrained(
                model,
                "lora",
                device_map=None,
            )
            model = model.merge_and_unload()
            logger.info("Loaded model with adapter weights")
        else:
            if lora_only:
                logger.error(
                    "Repo is not a lora weight, but lora_only flag is set to True."
                )
                raise InvalidModelException(
                    "Model is not LoRA but lora_only=True",
                    assignment_id=None,
                )
            logger.info("Repo is a full fine-tuned model, loading model directly")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, token=self.hf_token, **model_kwargs
            )

        if "output_router_logits" in model.config.to_dict():
            logger.info("set output_router_logits as True")
            model.config.output_router_logits = True

        logger.info(
            f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB"
        )

        total = sum(p.numel() for p in model.parameters())
        logger.info("Total model params: %.2fM" % (total / 1e6))

        return model

    def load_sft_dataset(
        self,
        eval_file: str,
        max_seq_length: int,
        template_name: str,
        tokenizer: AutoTokenizer,
    ) -> UnifiedSFTDataset:
        """
        Load a supervised fine-tuning dataset.

        Args:
            eval_file: Path to evaluation data file
            max_seq_length: Maximum sequence length
            template_name: Name of the chat template to use
            tokenizer: Tokenizer instance

        Returns:
            UnifiedSFTDataset instance

        Raises:
            InvalidDatasetException: If template or file is invalid
        """
        if template_name not in template_dict.keys():
            raise InvalidDatasetException(
                f"template_name doesn't exist, all template_name: {template_dict.keys()}",
                assignment_id=None,
            )
        template = template_dict[template_name]
        logger.info("Loading data with UnifiedSFTDataset")
        return UnifiedSFTDataset(eval_file, tokenizer, max_seq_length, template)

    def determine_tokenizer_path(
        self,
        model_name_or_path: str,
        revision: str,
    ) -> Tuple[str, bool]:
        """
        Determine the correct tokenizer path, handling LoRA models.

        Args:
            model_name_or_path: Model identifier or path
            revision: Git revision

        Returns:
            Tuple of (tokenizer_path, is_lora)

        Raises:
            InvalidLoraConfigException: If LoRA config is invalid
        """
        is_lora = download_lora_config(model_name_or_path, revision)
        tokenizer_model_path = model_name_or_path
        adapter_config_path = Path("lora/adapter_config.json")

        if is_lora:
            if not adapter_config_path.exists():
                raise InvalidLoraConfigException(
                    f"Model {model_name_or_path} is identified as LoRA, but adapter_config.json was not found",
                    assignment_id=None,
                )

            logger.info(
                f"Model {model_name_or_path} is a LoRA model. Validating its base model for tokenizer."
            )
            try:
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)

                lora_base_model_path = adapter_config.get("base_model_name_or_path")

                if not lora_base_model_path:
                    raise InvalidLoraConfigException(
                        f"LoRA model {model_name_or_path} does not specify 'base_model_name_or_path' in adapter_config.json",
                        assignment_id=None,
                    )

                if lora_base_model_path not in SUPPORTED_BASE_MODELS:
                    raise InvalidLoraConfigException(
                        f"LoRA's base model '{lora_base_model_path}' is not in SUPPORTED_BASE_MODELS",
                        assignment_id=None,
                    )

                logger.info(
                    f"LoRA's base model '{lora_base_model_path}' is in SUPPORTED_BASE_MODELS. Using it for tokenizer."
                )
                tokenizer_model_path = lora_base_model_path

            except json.JSONDecodeError as e:
                raise InvalidLoraConfigException(
                    f"Failed to decode adapter_config.json for {model_name_or_path}",
                    assignment_id=None,
                ) from e
            except Exception as e:
                if isinstance(e, InvalidLoraConfigException):
                    raise
                raise InvalidLoraConfigException(
                    f"Error processing adapter_config.json for {model_name_or_path}: {e}",
                    assignment_id=None,
                ) from e
        else:
            logger.info(
                f"Model {model_name_or_path} is not identified as a LoRA model. Using its own path for tokenizer."
            )

        return tokenizer_model_path, is_lora

    def evaluate_model(
        self,
        model_name_or_path: str,
        base_model: str,
        eval_file: str,
        context_length: int,
        max_params: int,
        val_args: TrainingArguments,
        assignment_id: Optional[str] = None,
        lora_only: bool = True,
        revision: str = "main",
    ) -> ValidationResult:
        """
        Execute a complete model evaluation workflow.

        Args:
            model_name_or_path: Model identifier or path
            base_model: Base model name for template selection
            eval_file: Path to evaluation dataset
            context_length: Maximum sequence length
            max_params: Maximum allowed model parameters
            val_args: Training arguments for evaluation
            assignment_id: Optional assignment ID for tracking
            lora_only: Whether to only accept LoRA models
            revision: Git revision to load

        Returns:
            ValidationResult with evaluation metrics

        Raises:
            ValidationException subclasses for specific error conditions
        """
        model = None
        eval_dataset = None
        
        try:
            tokenizer_model_path, cached_lora = self.determine_tokenizer_path(
                model_name_or_path, revision
            )

            tokenizer = self.load_tokenizer(tokenizer_model_path)
            eval_dataset = self.load_sft_dataset(
                eval_file, context_length, template_name=base_model, tokenizer=tokenizer
            )

            total_bytes, total_target_tokens = calculate_bytes_and_tokens(
                eval_dataset, tokenizer, logger
            )

            token_byte_ratio_value = get_token_byte_ratio(
                total_target_tokens, total_bytes
            )

            if total_bytes == 0:
                logger.warning(
                    "Total bytes in the evaluation dataset is 0. Cannot calculate BPC. Check dataset processing."
                )
            else:
                logger.info(f"Total target bytes (B): {total_bytes}")
                logger.info(f"Total target tokens (T): {total_target_tokens}")
                logger.info(f"Token/Byte ratio (T/B): {token_byte_ratio_value:.4f}")
                if token_byte_ratio_value < 0.1:
                    logger.warning(
                        f"Token/Byte ratio ({token_byte_ratio_value:.4f}) is unusually low. Potential manipulation detected."
                    )

            model = self.load_model(
                model_name_or_path, lora_only, revision, val_args, cached_lora
            )

            total = sum(p.numel() for p in model.parameters())
            if total > max_params:
                logger.error(
                    f"Total model params: {total} exceeds the limit {max_params}"
                )
                raise ModelParamsExceededException(
                    actual_params=total,
                    max_params=max_params,
                    assignment_id=assignment_id,
                )

            data_collator = SFTDataCollator(tokenizer, max_seq_length=context_length)

            trainer = Trainer(
                model=model,
                args=val_args,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            logger.info("Starting evaluation...")
            eval_result = trainer.evaluate()
            eval_loss = eval_result["eval_loss"]

            logger.info("Raw evaluation result: %s" % str(eval_result))

            bpc_metrics_results = calculate_bpc_bppl_metrics(
                eval_loss, total_target_tokens, total_bytes
            )

            is_bpc_valid = not math.isinf(bpc_metrics_results["bpc"])

            if is_bpc_valid:
                eval_loss_to_submit = bpc_metrics_results["bpc"]
            else:
                if (
                    not isinstance(eval_loss, numbers.Real)
                    or math.isnan(eval_loss)
                    or math.isinf(eval_loss)
                ):
                    logger.error(f"Invalid eval_loss ({eval_loss}), submitting high loss.")
                eval_loss_to_submit = LOSS_FOR_MODEL_PARAMS_EXCEED

            return ValidationResult(
                success=True,
                eval_loss=eval_loss,
                bpc=bpc_metrics_results["bpc"],
                bppl=bpc_metrics_results["bppl"],
                eval_loss_to_submit=eval_loss_to_submit,
                total_bytes=total_bytes,
                total_target_tokens=total_target_tokens,
                token_byte_ratio=token_byte_ratio_value,
                vocab_size=tokenizer.vocab_size,
                model_params_m=total / 1e6,
                assignment_id=assignment_id,
                metadata={
                    "nll_token_nats_total": bpc_metrics_results["nll_token_nats_total"],
                    "nll_token_bits_total": bpc_metrics_results["nll_token_bits_total"],
                },
            )

        finally:
            gc.collect()
            if model is not None:
                logger.debug("Offloading model to save memory")
                model.cpu()
                del model
            if eval_dataset is not None:
                logger.debug("Offloading eval_dataset to save memory")
                del eval_dataset
            torch.cuda.empty_cache()
            if os.path.exists("lora"):
                logger.debug("Removing lora folder")
                os.system("rm -rf lora")
