"""Pydantic schemas for the chat completions API."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator


class ChatMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional participant name")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier to use for generation")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = Field(False)
    max_tokens: Optional[int] = Field(None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    user: Optional[str] = None

    @validator("messages")
    def _validate_messages(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if not value:
            raise ValueError("messages must contain at least one item")
        if value[-1].role == ChatMessageRole.SYSTEM:
            raise ValueError("last message cannot be from system role")
        return value


class ChatMessageDelta(BaseModel):
    role: Optional[ChatMessageRole] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatMessageDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionStreamResponse",
    "ChatMessage",
    "ChatMessageRole",
]
