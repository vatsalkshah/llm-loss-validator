from fastapi.testclient import TestClient

from src.server.main import create_app


def get_client():
    app = create_app()
    return TestClient(app)


def test_healthz():
    client = get_client()
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_generate_completion():
    client = get_client()
    resp = client.post(
        "/v1/generate",
        json={
            "model": "echo-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"


def test_generate_stream():
    client = get_client()
    resp = client.post(
        "/v1/generate/stream",
        json={
            "model": "echo-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello world"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.text.strip().splitlines()
    assert body, "stream should yield chunks"
    assert body[-1].startswith("data: ")
