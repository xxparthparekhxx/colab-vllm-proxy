"""
Ollama-API-compatible proxy that fans out to multiple vLLM OpenAI-servers.

Exposes on ONE port (default 11434, same as Ollama):
  GET  /                 -> "Ollama is running"
  GET  /api/tags         -> list of "installed" models
  GET  /api/version      -> fake version string
  POST /api/generate     -> single-prompt completion (NDJSON stream or single JSON)
  POST /api/chat         -> chat completion (NDJSON stream or single JSON)
  POST /api/embeddings   -> legacy single-prompt embedding
  POST /api/embed        -> newer batch embedding endpoint

Internally forwards to whichever vLLM process owns that model, translating
request/response shapes both ways so your app never knows vLLM is back there.
"""

import json
import logging
import os
import secrets
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

logger = logging.getLogger("vllm-proxy")

app = FastAPI()

# ---------------------------------------------------------------------------
# Model routing table. Edit ports / HF ids here if you change your setup.
# "type" is "chat" (text or vision-chat models) or "embed".
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "gemma4": {
        "backend": "http://127.0.0.1:8001/v1",
        "hf_model": "google/gemma-4-E4B-it",
        "type": "chat",
    },
    "deepseek-ocr": {
        "backend": "http://127.0.0.1:8002/v1",
        "hf_model": "deepseek-ai/DeepSeek-OCR",
        "type": "chat",
        # DeepSeek-OCR needs these extra sampling params to avoid degenerate output
        "extra_body": {
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    },
    "nomic-embed-text": {
        "backend": "http://127.0.0.1:8003/v1",
        "hf_model": "nomic-ai/nomic-embed-text-v1.5",
        "type": "embed",
    },
}

client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0))


def resolve_model(name: str) -> Optional[dict]:
    """Ollama model names often carry a :tag suffix (e.g. gemma4:latest, gemma4:26b)."""
    base = name.split(":")[0]
    return MODEL_MAP.get(base) or MODEL_MAP.get(name)


# ---------------------------------------------------------------------------
# Error translation: Ollama clients expect {"error": "..."} JSON, never a
# plain-text FastAPI 500.
# ---------------------------------------------------------------------------

def ollama_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def upstream_status(status_code: int) -> int:
    """Keep 4xx (client's fault) and turn upstream 5xx into a 502 Bad Gateway."""
    if 400 <= status_code < 500:
        return status_code
    return 502


def error_message_from_body(status_code: int, body: bytes) -> str:
    detail = ""
    if body:
        try:
            parsed = json.loads(body)
        except (ValueError, TypeError):
            detail = body.decode("utf-8", "replace")
        else:
            err = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(err, dict):
                detail = err.get("message") or json.dumps(err)
            elif isinstance(err, str):
                detail = err
            else:
                detail = json.dumps(parsed)
    detail = (detail or "").strip()
    if len(detail) > 500:
        detail = detail[:500] + "..."
    message = f"upstream error {status_code}"
    return f"{message}: {detail}" if detail else message


def error_json_line(message: str) -> str:
    """One NDJSON error chunk, matching Ollama's in-stream error shape."""
    return json.dumps({"error": message}) + "\n"


def generate_done_line(model_name: str) -> str:
    return json.dumps({
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "response": "",
        "done": True,
    }) + "\n"


def chat_done_line(model_name: str) -> str:
    return json.dumps({
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }) + "\n"


def to_openai_messages(ollama_messages):
    """Convert Ollama chat 'images' (base64, no data-uri prefix) into OpenAI image_url blocks."""
    out = []
    for m in ollama_messages:
        content = m.get("content", "")
        images = m.get("images")
        if images:
            parts = [{"type": "text", "text": content}] if content else []
            for img_b64 in images:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            out.append({"role": m["role"], "content": parts})
        else:
            out.append({"role": m["role"], "content": content})
    return out


def ollama_options_to_openai(options: dict) -> dict:
    if not options:
        return {}
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "num_predict": "max_tokens",
        "seed": "seed",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "stop": "stop",
    }
    out = {}
    dropped = []
    for ok, ov in options.items():
        if ok in mapping:
            out[mapping[ok]] = ov
        else:
            dropped.append(ok)
    if dropped:
        logger.warning(
            "dropping Ollama option(s) with no OpenAI equivalent: %s",
            ", ".join(sorted(dropped)),
        )
    return out


# ---------------------------------------------------------------------------
# Optional bearer-token auth. Enabled only when PROXY_API_KEY is set.
# ---------------------------------------------------------------------------

def proxy_api_key() -> str:
    return os.environ.get("PROXY_API_KEY", "").strip()


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    api_key = proxy_api_key()
    if api_key:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token.encode("utf-8"), api_key.encode("utf-8")
        ):
            return ollama_error("unauthorized: missing or invalid API key", 401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return PlainTextResponse("Ollama is running")


@app.get("/api/version")
async def version():
    return {"version": "0.0.0-vllm-proxy"}


@app.get("/api/tags")
async def tags():
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    models = []
    for name, cfg in MODEL_MAP.items():
        models.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": now,
            "size": 0,
            "digest": "vllm-proxy",
            "details": {
                "family": name,
                "parameter_size": "",
                "quantization_level": "",
            },
        })
    return {"models": models}


# ---------------------------------------------------------------------------
# /api/generate  (single prompt -> completion)
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate(req: Request):
    body = await req.json()
    model_name = body["model"]
    cfg = resolve_model(model_name)
    if cfg is None:
        return JSONResponse({"error": f"model '{model_name}' not found"}, status_code=404)

    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    messages.append({"role": "user", "content": body.get("prompt", "")})
    if body.get("images"):
        parts = [{"type": "text", "text": body.get("prompt", "")}]
        for img_b64 in body["images"]:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        messages = messages[:-1] + [{"role": "user", "content": parts}]

    stream = body.get("stream", True)
    payload = {
        "model": cfg["hf_model"],
        "messages": messages,
        "stream": stream,
        **ollama_options_to_openai(body.get("options", {})),
    }
    if "extra_body" in cfg:
        payload.update(cfg["extra_body"])

    if stream:
        return StreamingResponse(
            stream_generate(cfg["backend"], payload, model_name),
            media_type="application/x-ndjson",
        )
    else:
        try:
            r = await client.post(f"{cfg['backend']}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            logger.warning("upstream request to %s failed: %s", cfg["backend"], exc)
            return ollama_error(f"upstream request failed: {exc}", 502)
        if r.status_code >= 400:
            return ollama_error(
                error_message_from_body(r.status_code, r.content),
                upstream_status(r.status_code),
            )
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("malformed upstream response from %s: %s", cfg["backend"], exc)
            return ollama_error("malformed response from upstream", 502)
        return JSONResponse({
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response": text,
            "done": True,
            "context": [],
            "eval_count": data.get("usage", {}).get("completion_tokens", 0),
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens", 0),
        })


async def stream_generate(backend, payload, model_name):
    done_sent = False
    try:
        async with client.stream("POST", f"{backend}/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                body = await r.aread()
                yield error_json_line(error_message_from_body(r.status_code, body))
                yield generate_done_line(model_name)
                return
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    yield generate_done_line(model_name)
                    done_sent = True
                    break
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    out = {
                        "model": model_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "response": delta,
                        "done": False,
                    }
                    yield json.dumps(out) + "\n"
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("stream from %s failed: %s", backend, exc)
        yield error_json_line(f"upstream stream failed: {exc}")
    if not done_sent:
        yield generate_done_line(model_name)


# ---------------------------------------------------------------------------
# /api/chat  (messages -> completion)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    model_name = body["model"]
    cfg = resolve_model(model_name)
    if cfg is None:
        return JSONResponse({"error": f"model '{model_name}' not found"}, status_code=404)

    messages = to_openai_messages(body.get("messages", []))
    stream = body.get("stream", True)
    payload = {
        "model": cfg["hf_model"],
        "messages": messages,
        "stream": stream,
        **ollama_options_to_openai(body.get("options", {})),
    }
    if "extra_body" in cfg:
        payload.update(cfg["extra_body"])

    if stream:
        return StreamingResponse(
            stream_chat(cfg["backend"], payload, model_name),
            media_type="application/x-ndjson",
        )
    else:
        try:
            r = await client.post(f"{cfg['backend']}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            logger.warning("upstream request to %s failed: %s", cfg["backend"], exc)
            return ollama_error(f"upstream request failed: {exc}", 502)
        if r.status_code >= 400:
            return ollama_error(
                error_message_from_body(r.status_code, r.content),
                upstream_status(r.status_code),
            )
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("malformed upstream response from %s: %s", cfg["backend"], exc)
            return ollama_error("malformed response from upstream", 502)
        return JSONResponse({
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": {"role": "assistant", "content": text},
            "done": True,
            "eval_count": data.get("usage", {}).get("completion_tokens", 0),
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens", 0),
        })


async def stream_chat(backend, payload, model_name):
    done_sent = False
    try:
        async with client.stream("POST", f"{backend}/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                body = await r.aread()
                yield error_json_line(error_message_from_body(r.status_code, body))
                yield chat_done_line(model_name)
                return
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    yield chat_done_line(model_name)
                    done_sent = True
                    break
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    out = {
                        "model": model_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "message": {"role": "assistant", "content": delta},
                        "done": False,
                    }
                    yield json.dumps(out) + "\n"
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("stream from %s failed: %s", backend, exc)
        yield error_json_line(f"upstream stream failed: {exc}")
    if not done_sent:
        yield chat_done_line(model_name)


# ---------------------------------------------------------------------------
# /api/embeddings (legacy) and /api/embed (batch)
# ---------------------------------------------------------------------------

@app.post("/api/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    model_name = body["model"]
    cfg = resolve_model(model_name)
    if cfg is None:
        return JSONResponse({"error": f"model '{model_name}' not found"}, status_code=404)

    payload = {"model": cfg["hf_model"], "input": body.get("prompt", "")}
    try:
        r = await client.post(f"{cfg['backend']}/embeddings", json=payload)
    except httpx.HTTPError as exc:
        logger.warning("upstream request to %s failed: %s", cfg["backend"], exc)
        return ollama_error(f"upstream request failed: {exc}", 502)
    if r.status_code >= 400:
        return ollama_error(
            error_message_from_body(r.status_code, r.content),
            upstream_status(r.status_code),
        )
    try:
        data = r.json()
        embedding = data["data"][0]["embedding"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("malformed upstream response from %s: %s", cfg["backend"], exc)
        return ollama_error("malformed response from upstream", 502)
    return JSONResponse({"embedding": embedding})


@app.post("/api/embed")
async def embed(req: Request):
    body = await req.json()
    model_name = body["model"]
    cfg = resolve_model(model_name)
    if cfg is None:
        return JSONResponse({"error": f"model '{model_name}' not found"}, status_code=404)

    inputs = body.get("input", "")
    payload = {"model": cfg["hf_model"], "input": inputs}
    try:
        r = await client.post(f"{cfg['backend']}/embeddings", json=payload)
    except httpx.HTTPError as exc:
        logger.warning("upstream request to %s failed: %s", cfg["backend"], exc)
        return ollama_error(f"upstream request failed: {exc}", 502)
    if r.status_code >= 400:
        return ollama_error(
            error_message_from_body(r.status_code, r.content),
            upstream_status(r.status_code),
        )
    try:
        data = r.json()
        vectors = [item["embedding"] for item in data["data"]]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("malformed upstream response from %s: %s", cfg["backend"], exc)
        return ollama_error("malformed response from upstream", 502)
    return JSONResponse({"model": model_name, "embeddings": vectors})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Bind loopback by default; cloudflared runs on the same host. Set HOST=0.0.0.0
    # only if you really want the port reachable from the network.
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=11434)
