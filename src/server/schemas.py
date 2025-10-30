"""
Pydantic schemas for the inference API server.

These schemas mirror the OpenAI Chat Completions API to provide
compatibility for existing clients while leveraging FastAPI's
validation features.
"""

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
    name: Optional[str] = Field(None, description="Name associated with the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for generation")
    messages: List[ChatMessage] = Field(
        ..., description="List of messages leading up to the request"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    stream: bool = Field(False, description="Whether to stream responses")
    max_tokens: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum tokens to generate"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Stop sequences"
    )
    presence_penalty: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    user: Optional[str] = Field(None, description="End-user identifier")

    @validator("messages")
    def validate_messages(cls, messages: List[ChatMessage]) -> List[ChatMessage]:
        if not messages:
            raise ValueError("messages must contain at least one message")
        if messages[-1].role == ChatMessageRole.SYSTEM:
            raise ValueError("Last message cannot be from system role")
        return messages


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
