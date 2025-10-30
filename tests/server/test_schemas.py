"""
Tests for request/response schemas.
"""

import pytest
from pydantic import ValidationError

from server.schemas import (
    ChatMessage,
    ChatMessageRole,
    ChatCompletionRequest,
)


def test_chat_message_valid():
    """Test valid chat message creation."""
    msg = ChatMessage(role=ChatMessageRole.USER, content="Hello")
    assert msg.role == ChatMessageRole.USER
    assert msg.content == "Hello"
    assert msg.name is None


def test_chat_completion_request_valid():
    """Test valid chat completion request."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")],
        temperature=0.8,
        max_tokens=100,
    )
    assert req.model == "test-model"
    assert len(req.messages) == 1
    assert req.temperature == 0.8
    assert req.max_tokens == 100
    assert req.stream is False


def test_chat_completion_request_empty_messages():
    """Test that empty messages list is rejected."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[],
        )


def test_chat_completion_request_last_message_system():
    """Test that system as last message is rejected."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role=ChatMessageRole.USER, content="Hello"),
                ChatMessage(role=ChatMessageRole.SYSTEM, content="System message"),
            ],
        )


def test_chat_completion_request_temperature_range():
    """Test temperature validation."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")],
            temperature=3.0,
        )

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")],
            temperature=-1.0,
        )


def test_chat_completion_request_top_p_range():
    """Test top_p validation."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")],
            top_p=1.5,
        )


def test_chat_completion_request_max_tokens_positive():
    """Test max_tokens must be positive."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")],
            max_tokens=0,
        )


def test_chat_completion_request_streaming():
    """Test streaming flag."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")],
        stream=True,
    )
    assert req.stream is True
