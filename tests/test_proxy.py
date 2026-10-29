import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import proxy as proxy_module


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Auth is off unless a test turns it on."""
    monkeypatch.delenv("PROXY_API_KEY", raising=False)


@pytest.fixture
def client():
    with TestClient(proxy_module.app) as c:
        yield c


@pytest.fixture
def fake_upstream(monkeypatch):
    """Install an httpx.MockTransport as the proxy's upstream client."""

    def install(handler):
        requests = []

        def wrapped(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(
            proxy_module,
            "client",
            httpx.AsyncClient(transport=httpx.MockTransport(wrapped)),
        )
        return requests

    return install


def chat_body(content="hello"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }


def ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class ExplodingStream(httpx.AsyncByteStream):
    """Yields one SSE chunk, then dies mid-stream."""

    async def __aiter__(self):
        chunk = {"choices": [{"delta": {"content": "partial"}}]}
        yield ("data: " + json.dumps(chunk) + "\n\n").encode()
        raise httpx.ReadError("simulated connection reset")


def test_non_streaming_upstream_500_is_translated(client, fake_upstream):
    fake_upstream(
        lambda request: httpx.Response(500, json={"error": {"message": "CUDA out of memory"}})
    )

    r = client.post("/api/generate", json={"model": "gemma4", "prompt": "hi", "stream": False})

    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert "CUDA out of memory" in r.json()["error"]


def test_non_streaming_upstream_4xx_passes_through(client, fake_upstream):
    fake_upstream(lambda request: httpx.Response(400, json={"error": "bad request"}))

    r = client.post(
        "/api/chat",
        json={"model": "gemma4", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )

    assert r.status_code == 400
    assert "bad request" in r.json()["error"]


def test_embed_upstream_error_is_translated(client, fake_upstream):
    fake_upstream(lambda request: httpx.Response(503, text="backend down"))

    r = client.post("/api/embed", json={"model": "nomic-embed-text", "input": "hi"})

    assert r.status_code == 502
    assert "backend down" in r.json()["error"]


def test_upstream_connection_error_is_translated(client, fake_upstream):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    fake_upstream(handler)

    r = client.post("/api/generate", json={"model": "gemma4", "prompt": "hi", "stream": False})

    assert r.status_code == 502
    assert "connection refused" in r.json()["error"]


def test_stream_midstream_failure_ends_with_done(client, fake_upstream):
    fake_upstream(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ExplodingStream(),
        )
    )

    r = client.post("/api/generate", json={"model": "gemma4", "prompt": "hi", "stream": True})

    chunks = ndjson(r.text)
    assert chunks[0]["response"] == "partial"
    assert any("error" in c for c in chunks)
    assert chunks[-1]["done"] is True
    assert "simulated connection reset" in chunks[-2]["error"]


def test_chat_stream_midstream_failure_ends_with_done(client, fake_upstream):
    fake_upstream(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ExplodingStream(),
        )
    )

    r = client.post(
        "/api/chat",
        json={"model": "gemma4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    chunks = ndjson(r.text)
    assert chunks[0]["message"]["content"] == "partial"
    assert any("error" in c for c in chunks)
    assert chunks[-1]["done"] is True


def test_stream_upstream_500_yields_error_and_done(client, fake_upstream):
    fake_upstream(lambda request: httpx.Response(500, json={"error": "model crashed"}))

    r = client.post("/api/generate", json={"model": "gemma4", "prompt": "hi", "stream": True})

    chunks = ndjson(r.text)
    assert len(chunks) >= 2
    assert any("model crashed" in c.get("error", "") for c in chunks)
    assert chunks[-1]["done"] is True
    # No partial content chunks: the upstream failed before producing any.
    assert all(c.get("response", "") == "" for c in chunks)


def test_generate_maps_system_prompt(client, fake_upstream):
    requests = fake_upstream(lambda request: httpx.Response(200, json=chat_body()))

    r = client.post(
        "/api/generate",
        json={"model": "gemma4", "system": "You are terse.", "prompt": "hi", "stream": False},
    )

    assert r.status_code == 200
    payload = json.loads(requests[0].content)
    assert payload["messages"][0] == {"role": "system", "content": "You are terse."}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


def test_dropped_options_are_logged(client, fake_upstream, caplog):
    requests = fake_upstream(lambda request: httpx.Response(200, json=chat_body()))

    with caplog.at_level(logging.WARNING, logger="vllm-proxy"):
        r = client.post(
            "/api/generate",
            json={
                "model": "gemma4",
                "prompt": "hi",
                "stream": False,
                "options": {"top_k": 40, "num_ctx": 4096, "temperature": 0.2},
            },
        )

    assert r.status_code == 200
    assert "top_k" in caplog.text
    assert "num_ctx" in caplog.text
    payload = json.loads(requests[0].content)
    assert payload["temperature"] == 0.2
    assert "top_k" not in payload
    assert "num_ctx" not in payload


def test_auth_disabled_by_default(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/tags").status_code == 200


def test_auth_requires_bearer_when_configured(client, monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "s3cret")

    missing = client.get("/")
    assert missing.status_code == 401
    assert "error" in missing.json()

    wrong = client.get("/", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401

    wrong_scheme = client.get("/", headers={"Authorization": "Basic s3cret"})
    assert wrong_scheme.status_code == 401

    ok = client.get("/", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_auth_covers_post_routes(client, monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "s3cret")

    r = client.post("/api/generate", json={"model": "gemma4", "prompt": "hi"})
    assert r.status_code == 401
    assert "error" in r.json()


# ---------------------------------------------------------------------------
# Host resolution (no real server started)
# ---------------------------------------------------------------------------

def test_resolve_host_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    assert proxy_module.resolve_host() == "127.0.0.1"


def test_resolve_host_empty_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("HOST", "")
    assert proxy_module.resolve_host() == "127.0.0.1"


def test_resolve_host_uses_set_value(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert proxy_module.resolve_host() == "0.0.0.0"

    monkeypatch.setenv("HOST", "127.0.0.1")
    assert proxy_module.resolve_host() == "127.0.0.1"


# ---------------------------------------------------------------------------
# /api/chat stream=true upstream 500
# ---------------------------------------------------------------------------

def test_chat_stream_upstream_500_yields_error_and_done(client, fake_upstream):
    fake_upstream(lambda request: httpx.Response(500, json={"error": "model crashed"}))

    r = client.post(
        "/api/chat",
        json={"model": "gemma4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert r.headers["content-type"].startswith("application/x-ndjson")
    chunks = ndjson(r.text)
    assert len(chunks) >= 2
    assert any("model crashed" in c.get("error", "") for c in chunks)
    assert chunks[-1]["done"] is True
    # No partial content chunks: the upstream failed before producing any.
    assert all(c.get("message", {}).get("content", "") == "" for c in chunks)


# ---------------------------------------------------------------------------
# Malformed requests -> 400 JSON (never a plain-text 500)
# ---------------------------------------------------------------------------

def test_generate_invalid_json_returns_400_json(client):
    r = client.post(
        "/api/generate",
        content="not json at all",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "error" in r.json()
    assert "invalid JSON" in r.json()["error"]


def test_generate_missing_model_returns_400_json(client):
    r = client.post("/api/generate", json={"prompt": "hi"})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "error" in r.json()
    assert "model" in r.json()["error"].lower()


def test_chat_invalid_json_returns_400_json(client):
    r = client.post(
        "/api/chat",
        content="{",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "error" in r.json()


def test_chat_missing_model_returns_400_json(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "error" in r.json()
    assert "model" in r.json()["error"].lower()


# ---------------------------------------------------------------------------
# Whitespace-only key refuses to start (fail closed)
# ---------------------------------------------------------------------------

def test_blank_api_key_refuses_to_start(monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "   ")
    with pytest.raises(SystemExit) as exc:
        proxy_module.require_valid_api_key()
    assert exc.value.code
    assert "PROXY_API_KEY" in str(exc.value)


def test_unset_api_key_does_not_refuse_to_start(monkeypatch):
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    proxy_module.require_valid_api_key()  # must not raise


def test_valid_api_key_does_not_refuse_to_start(monkeypatch):
    monkeypatch.setenv("PROXY_API_KEY", "s3cret")
    proxy_module.require_valid_api_key()  # must not raise
