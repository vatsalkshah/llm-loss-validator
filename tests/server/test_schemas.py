import pytest

from src.server.schemas import ChatCompletionRequest, ChatMessage, ChatMessageRole


def test_valid_request():
    req = ChatCompletionRequest(
        model="echo-model",
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content="system"),
            ChatMessage(role=ChatMessageRole.USER, content="hi"),
        ],
    )
    assert req.model == "echo-model"


def test_validator_disallows_empty_messages():
    with pytest.raises(ValueError):
        ChatCompletionRequest(model="x", messages=[])


def test_validator_disallows_last_system():
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            model="x",
            messages=[
                ChatMessage(role=ChatMessageRole.USER, content="hi"),
                ChatMessage(role=ChatMessageRole.SYSTEM, content="bye"),
            ],
        )
