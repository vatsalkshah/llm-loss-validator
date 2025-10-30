import json
import time
from typing import Iterator, Dict

import pytest
from fastapi.testclient import TestClient


class StubCacheManager:
    def __init__(self) -> None:
        self.cached_models = {"stub-model"}
        self.active_models = set()

    def list_models(self, include_inactive: bool = True):
        data = []
        for model_id in self.cached_models:
            data.append(
                {
                    "model_id": model_id,
                    "model_type": "base",
                    "size_bytes": 0,
                    "download_timestamp": time.time(),
                    "last_accessed": time.time(),
                    "revision": "main",
                    "cache_path": f"/fake/{model_id}",
                    "is_active": model_id in self.active_models,
                }
            )
        return data

    def is_model_cached(self, model_id: str, revision: str = "main") -> bool:
        return model_id in self.cached_models

    def mark_active(self, model_id: str, revision: str = "main", active: bool = True) -> bool:
        if active:
            self.active_models.add(model_id)
        else:
            self.active_models.discard(model_id)
        return True


class StubModelLoader:
    def __init__(self) -> None:
        self.loaded_models = {}

    def load_model(self, model_id: str, revision: str = "main", force_reload: bool = False):
        self.loaded_models[model_id] = revision
        return object(), object()

    def list_loaded_models(self):
        return [f"{model_id}@{revision}" for model_id, revision in self.loaded_models.items()]

    def unload_model(self, model_id: str, revision: str = "main") -> bool:
        self.loaded_models.pop(model_id, None)
        return True


class StubTextGenerator:
    def __init__(self, model, tokenizer, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.generate_calls = []
        self.generate_stream_calls = []

    def generate(self, **kwargs) -> Dict:
        self.generate_calls.append(kwargs)
        return {
            "text": "stub-response",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }

    def generate_stream(self, **kwargs) -> Iterator[Dict]:
        self.generate_stream_calls.append(kwargs)
        yield {
            "text": "stub",
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        }
        yield {
            "text": "-response",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }


@pytest.fixture
def api_client(monkeypatch):
    from server import main

    cache_manager = StubCacheManager()
    model_loader = StubModelLoader()
    generator_instances = []

    def stub_model_cache_manager(*args, **kwargs):
        return cache_manager

    def stub_model_loader(*args, **kwargs):
        return model_loader

    def stub_text_generator(*args, **kwargs):
        generator = StubTextGenerator(*args, **kwargs)
        generator_instances.append(generator)
        return generator

    monkeypatch.setattr(main, "ModelCacheManager", stub_model_cache_manager)
    monkeypatch.setattr(main, "ModelLoader", stub_model_loader)
    monkeypatch.setattr(main, "TextGenerator", stub_text_generator)
    monkeypatch.setattr(main.config, "require_api_key", False)
    monkeypatch.setattr(main.config, "reject_non_cached_models", False)

    with TestClient(main.app) as client:
        yield client, cache_manager, model_loader, generator_instances


def test_chat_completions_non_stream(api_client):
    client, cache_manager, model_loader, generators = api_client

    payload = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "stub-response"
    assert data["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }

    # Ensure generator was used
    assert generators
    assert generators[0].generate_calls


def test_chat_completions_stream(api_client):
    client, cache_manager, model_loader, generators = api_client

    payload = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        chunks = [line for line in response.iter_lines() if line]

    # Expect first chunk with role, two content chunks, and DONE marker
    assert chunks[-1] == "data: [DONE]"

    # Parse middle chunks
    content_chunks = chunks[1:-1]
    aggregated = ""
    for chunk_line in content_chunks:
        assert chunk_line.startswith("data: ")
        payload = json.loads(chunk_line[len("data: "):])
        assert payload["object"] == "chat.completion.chunk"
        delta = payload["choices"][0]["delta"]
        aggregated += delta.get("content", "")

    assert aggregated == "stub-response"
    assert generators
    assert generators[-1].generate_stream_calls


def test_rejects_non_cached_model(api_client, monkeypatch):
    client, cache_manager, model_loader, generators = api_client

    # Configure cache to have no models and enable rejection
    cache_manager.cached_models.clear()
    from server import main

    monkeypatch.setattr(main.config, "reject_non_cached_models", True)

    payload = {
        "model": "missing-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "not cached" in data["detail"].lower()
