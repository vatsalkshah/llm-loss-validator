import httpx
import pytest

from src.core.telemetry import TelemetryClient, TelemetryError


def test_emit_success(monkeypatch):
    captured = {}

    class DummyTransport(httpx.BaseTransport):
        def handle_request(self, request):  # pragma: no cover - exercised indirectly
            captured["json"] = httpx.Request("POST", request.url, content=request.content).json()
            return httpx.Response(200, json={"ok": True})

    client = TelemetryClient("https://example.com/telemetry", client=httpx.Client(transport=DummyTransport()))
    resp = client.emit("test", {"value": 1})
    assert resp.status_code == 200
    assert captured["json"]["event"] == "test"


def test_emit_failure(monkeypatch):
    class DummyTransport(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(500)

    client = TelemetryClient("https://example.com/telemetry", client=httpx.Client(transport=DummyTransport()))
    with pytest.raises(TelemetryError):
        client.emit("fail", {})


def test_noop_when_disabled():
    client = TelemetryClient(None)
    assert client.emit("noop", {"value": 1}) is None
