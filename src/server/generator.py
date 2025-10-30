"""High-level generation utilities that wrap a model loader."""
from __future__ import annotations

import time
import uuid
from typing import Dict, Iterable

from fastapi import HTTPException

from .model_loader import ModelLoader
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatMessageDelta,
    ChatMessageRole,
    ChatCompletionChoice,
    ChatCompletionUsage,
)


def _render_prompt(request: ChatCompletionRequest) -> str:
    parts = []
    for message in request.messages:
        parts.append(f"[{message.role}] {message.content}")
    return "\n".join(parts)


class GenerationPipeline:
    """Simple text generation pipeline used by the FastAPI endpoints."""

    def __init__(self, loader: ModelLoader) -> None:
        self.loader = loader

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Use the streaming endpoint for stream=true requests")

        model = self.loader.get(request.model)
        prompt = _render_prompt(request)
        completion = model.generate(prompt, max_tokens=request.max_tokens or 256)

        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role=ChatMessageRole.ASSISTANT, content=completion),
            finish_reason="stop",
        )
        usage = ChatCompletionUsage(prompt_tokens=len(prompt.split()), completion_tokens=len(completion.split()), total_tokens=len(prompt.split()) + len(completion.split()))

        return ChatCompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=usage,
        )

    def generate_stream(self, request: ChatCompletionRequest) -> Iterable[ChatCompletionStreamResponse]:
        model = self.loader.get(request.model)
        prompt = _render_prompt(request)
        completion = model.generate(prompt, max_tokens=request.max_tokens or 256)
        created = int(time.time())
        chunk_id = f"cmpl-{uuid.uuid4().hex}"

        for idx, token in enumerate(completion.split()):
            delta = ChatMessageDelta(role=ChatMessageRole.ASSISTANT if idx == 0 else None, content=token if token else None)
            yield ChatCompletionStreamResponse(
                id=chunk_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(index=0, delta=delta, finish_reason=None)],
            )

        # final chunk to signal completion
        yield ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionStreamChoice(index=0, delta=ChatMessageDelta(content=""), finish_reason="stop")],
        )


__all__ = ["GenerationPipeline"]
