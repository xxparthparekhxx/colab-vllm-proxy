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
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

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
    for ok, ov in options.items():
        if ok in mapping:
            out[mapping[ok]] = ov
    return out


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

    messages = [{"role": "user", "content": body.get("prompt", "")}]
    if body.get("images"):
        parts = [{"type": "text", "text": body.get("prompt", "")}]
        for img_b64 in body["images"]:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        messages = [{"role": "user", "content": parts}]

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
        r = await client.post(f"{cfg['backend']}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
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
    async with client.stream("POST", f"{backend}/chat/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if raw == "[DONE]":
                final = {
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "response": "",
                    "done": True,
                }
                yield json.dumps(final) + "\n"
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
        r = await client.post(f"{cfg['backend']}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return JSONResponse({
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": {"role": "assistant", "content": text},
            "done": True,
            "eval_count": data.get("usage", {}).get("completion_tokens", 0),
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens", 0),
        })


async def stream_chat(backend, payload, model_name):
    async with client.stream("POST", f"{backend}/chat/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if raw == "[DONE]":
                final = {
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
                yield json.dumps(final) + "\n"
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
    r = await client.post(f"{cfg['backend']}/embeddings", json=payload)
    r.raise_for_status()
    data = r.json()
    return JSONResponse({"embedding": data["data"][0]["embedding"]})


@app.post("/api/embed")
async def embed(req: Request):
    body = await req.json()
    model_name = body["model"]
    cfg = resolve_model(model_name)
    if cfg is None:
        return JSONResponse({"error": f"model '{model_name}' not found"}, status_code=404)

    inputs = body.get("input", "")
    payload = {"model": cfg["hf_model"], "input": inputs}
    r = await client.post(f"{cfg['backend']}/embeddings", json=payload)
    r.raise_for_status()
    data = r.json()
    vectors = [item["embedding"] for item in data["data"]]
    return JSONResponse({"model": model_name, "embeddings": vectors})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)