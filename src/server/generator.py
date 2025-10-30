"""
Text generation module for the inference API server.

This module handles prompt construction, generation parameter mapping,
and both streaming and non-streaming text generation using transformers.
"""

import uuid
import time
from typing import List, Dict, Optional, Iterator, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from server.schemas import ChatMessage, ChatMessageRole
from core.template import template_dict, Template


def format_chat_prompt(
    messages: List[ChatMessage],
    tokenizer: AutoTokenizer,
    template: Optional[Template] = None,
) -> str:
    """
    Format a list of chat messages into a prompt string using the appropriate template.
    
    Args:
        messages: List of chat messages
        tokenizer: Tokenizer instance
        template: Optional template override (defaults to 'default')
    
    Returns:
        Formatted prompt string
    """
    if template is None:
        template = template_dict.get("default")
    
    prompt_parts = []
    
    # Find system message if any
    system_message = None
    conversation_messages = []
    
    for msg in messages:
        if msg.role == ChatMessageRole.SYSTEM:
            system_message = msg.content
        else:
            conversation_messages.append(msg)
    
    # Add system message if present and template supports it
    if system_message and template.system_format:
        prompt_parts.append(template.system_format.format(content=system_message))
    elif template.system and template.system_format:
        # Use default system message from template if defined
        prompt_parts.append(template.system_format.format(content=template.system))
    
    # Format conversation messages
    for msg in conversation_messages:
        if msg.role == ChatMessageRole.USER:
            prompt_parts.append(template.user_format.format(content=msg.content))
        elif msg.role == ChatMessageRole.ASSISTANT:
            # For assistant messages in context, format with stop token
            stop_token = template.stop_word if template.stop_word else tokenizer.eos_token
            formatted = template.assistant_format.format(
                content=msg.content,
                stop_token=stop_token
            )
            prompt_parts.append(formatted)
        elif msg.role == ChatMessageRole.TOOL:
            prompt_parts.append(template.observation_format.format(content=msg.content))
        elif msg.role == ChatMessageRole.FUNCTION:
            prompt_parts.append(template.function_format.format(content=msg.content))
    
    return "".join(prompt_parts)


def detect_template_from_model(model_id: str) -> Optional[Template]:
    """
    Auto-detect the appropriate chat template based on model ID.
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Template instance or None if not detected
    """
    model_lower = model_id.lower()
    
    if "llama-3" in model_lower or "llama3" in model_lower:
        return template_dict.get("llama3")
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return template_dict.get("llama2")
    elif "qwen" in model_lower or "qwen2" in model_lower:
        return template_dict.get("qwen1.5")
    elif "yi" in model_lower:
        return template_dict.get("yi")
    elif "mistral" in model_lower:
        return template_dict.get("mistral")
    elif "mixtral" in model_lower:
        return template_dict.get("mixtral")
    elif "gemma" in model_lower:
        return template_dict.get("gemma")
    elif "phi-3" in model_lower:
        return template_dict.get("phi3")
    elif "phi-4" in model_lower or "phi4" in model_lower:
        return template_dict.get("phi4")
    elif "zephyr" in model_lower:
        return template_dict.get("zephyr")
    
    return template_dict.get("default")


class TextGenerator:
    """
    Handles text generation for chat completions.
    
    This class provides both streaming and non-streaming generation
    with support for various generation parameters.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_id: str,
    ):
        """
        Initialize the TextGenerator.
        
        Args:
            model: Loaded model instance
            tokenizer: Tokenizer instance
            model_id: Model identifier for template detection
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.template = detect_template_from_model(model_id)
        
        logger.info(f"TextGenerator initialized with template: {self.template.template_name if self.template else 'default'}")
    
    def _prepare_generation_kwargs(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Dict:
        """
        Prepare generation parameters for transformers generation.
        
        Args:
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
        
        Returns:
            Dictionary of generation kwargs
        """
        kwargs = {
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature is not None:
            kwargs["temperature"] = max(temperature, 0.01)  # Avoid 0 temperature
        else:
            kwargs["temperature"] = 0.7
        
        if top_p is not None:
            kwargs["top_p"] = top_p
        
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        else:
            kwargs["max_new_tokens"] = 512
        
        # Handle stop sequences
        stop_token_ids = []
        if stop:
            stop_list = [stop] if isinstance(stop, str) else stop
            for stop_seq in stop_list:
                # Try to encode stop sequence
                encoded = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if encoded:
                    stop_token_ids.extend(encoded)
        
        # Add template stop token if available
        if self.template and self.template.stop_word:
            template_stop = self.tokenizer.encode(
                self.template.stop_word,
                add_special_tokens=False
            )
            if template_stop:
                stop_token_ids.extend(template_stop)
        
        if stop_token_ids:
            # Remove duplicates
            kwargs["eos_token_id"] = list(set(stop_token_ids + [self.tokenizer.eos_token_id]))
        
        # Transformers doesn't directly support presence/frequency penalties
        # These would need custom logits processors, so we log and skip for now
        if presence_penalty is not None:
            logger.debug(f"presence_penalty={presence_penalty} requested but not yet implemented")
        if frequency_penalty is not None:
            logger.debug(f"frequency_penalty={frequency_penalty} requested but not yet implemented")
        
        return kwargs
    
    def generate(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Dict:
        """
        Generate a non-streaming completion.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
        
        Returns:
            Dictionary with generated text and token counts
        """
        # Format prompt
        prompt = format_chat_prompt(messages, self.tokenizer, self.template)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = self._prepare_generation_kwargs(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        
        # Generate
        logger.debug(f"Generating with kwargs: {gen_kwargs}")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the generated tokens (excluding prompt)
        generated_tokens = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completion_tokens = len(generated_tokens)
        
        return {
            "text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    
    def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Iterator[Dict]:
        """
        Generate a streaming completion.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
        
        Yields:
            Dictionaries with incremental text chunks
        """
        # Format prompt
        prompt = format_chat_prompt(messages, self.tokenizer, self.template)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = self._prepare_generation_kwargs(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer
        
        # Start generation in a separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs}
        )
        generation_thread.start()
        
        # Stream tokens
        completion_tokens = 0
        for text_chunk in streamer:
            if not text_chunk:
                continue
            chunk_token_ids = self.tokenizer(
                text_chunk,
                add_special_tokens=False,
                return_tensors=None
            )["input_ids"]
            chunk_token_count = len(chunk_token_ids)
            completion_tokens += chunk_token_count
            yield {
                "text": text_chunk,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        
        # Wait for generation to complete
        generation_thread.join()
