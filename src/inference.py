from __future__ import annotations

import asyncio
from threading import Thread
from time import perf_counter
from typing import Any, AsyncGenerator, Dict, Tuple

import torch
from loguru import logger
from transformers import TextIteratorStreamer

from src.core.model_registry import get_registry
from src.metrics import ThroughputMeter


def _prepare_inputs(tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        return {k: v.to("cuda") for k, v in inputs.items()}
    return inputs


async def stream_generate(model_id: str, prompt: str, gen_kwargs: Dict[str, Any]) -> AsyncGenerator[str, None]:
    registry = get_registry()
    model, tokenizer = registry.get(model_id)

    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    inputs = _prepare_inputs(tokenizer, prompt)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    default_gen = dict(max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.95)
    generate_kwargs = {**default_gen, **(gen_kwargs or {}), "inputs": inputs["input_ids"], "streamer": streamer}

    meter = ThroughputMeter()

    def _generate():
        with torch.inference_mode():
            model.generate(**generate_kwargs)

    thread = Thread(target=_generate)
    thread.start()

    token_idx = 0
    for text in streamer:
        token_idx += 1
        meter.tick(1)
        yield text

    thread.join()


def run_generate(model_id: str, prompt: str, gen_kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    registry = get_registry()
    model, tokenizer = registry.get(model_id)

    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    default_gen = dict(max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.95)
    generate_kwargs = {**default_gen, **(gen_kwargs or {}), "inputs": inputs["input_ids"]}

    start = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)
    elapsed = max(perf_counter() - start, 1e-6)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    usage = {"elapsed_sec": elapsed, "tokens_out": int(output_ids.shape[-1])}
    return text, usage
