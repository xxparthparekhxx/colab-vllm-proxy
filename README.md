# colab-vllm-proxy

An [Ollama](https://ollama.com)-API-compatible proxy that fans out to three
vLLM OpenAI-style servers. Your app talks Ollama (`/api/generate`, `/api/chat`,
`/api/embed`, ...) and never has to know vLLM is behind it.

| Model name        | Backend                | Port | Type  |
| ----------------- | ---------------------- | ---- | ----- |
| `gemma4`          | `google/gemma-4-E4B-it` | 8001 | chat  |
| `deepseek-ocr`    | `deepseek-ai/DeepSeek-OCR` | 8002 | chat  |
| `nomic-embed-text`| `nomic-ai/nomic-embed-text-v1.5` | 8003 | embed |

Model names accept an Ollama-style tag suffix (`gemma4:latest` resolves to
`gemma4`).

## Endpoints

| Method | Path             | Description                                   |
| ------ | ---------------- | --------------------------------------------- |
| GET    | `/`              | `Ollama is running`                           |
| GET    | `/api/version`   | Fake version string                           |
| GET    | `/api/tags`      | List of "installed" models                    |
| POST   | `/api/generate`  | Single-prompt completion (NDJSON or JSON)     |
| POST   | `/api/chat`      | Chat completion (NDJSON or JSON)              |
| POST   | `/api/embeddings`| Legacy single-prompt embedding                |
| POST   | `/api/embed`     | Batch embedding                               |

Upstream failures are translated into Ollama's `{"error": "..."}` JSON shape:
upstream 4xx statuses pass through, upstream 5xx becomes `502 Bad Gateway`.
A failure in the middle of a stream ends the NDJSON stream with an error line
and a final `{"done": true}` line, so clients are never left hanging on an
empty stream.

## Run

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` starts the three vLLM servers (bound to `127.0.0.1`), waits for them
to become ready, then starts the proxy. Each vLLM server runs under a
supervisor that restarts it with exponential backoff if it crashes.

Readiness is bounded: if a backend is not ready within `READY_TIMEOUT`
seconds, `run.sh` prints the tail of that backend's log and exits non-zero.

### Environment variables

| Variable                | Default     | Meaning                                             |
| ----------------------- | ----------- | --------------------------------------------------- |
| `READY_TIMEOUT`         | `900`       | Seconds to wait for each backend to become ready    |
| `RESTART_BACKOFF_START` | `5`         | First restart delay (seconds)                       |
| `RESTART_BACKOFF_MAX`   | `300`       | Maximum restart delay (seconds)                     |
| `HOST`                  | `127.0.0.1` | Proxy bind address                                  |
| `PROXY_API_KEY`         | unset       | If set, require `Authorization: Bearer <key>`       |
| `LOG_LEVEL`             | `INFO`      | Python logging level                                |

Example: `READY_TIMEOUT=1800 PROXY_API_KEY=... ./run.sh`

## Security

The proxy has no built-in user system. Two controls matter:

1. **Bind loopback.** The proxy listens on `127.0.0.1:11434` by default, not
   `0.0.0.0`. Only a process on the same host (normally `cloudflared`) can
   reach it. Set `HOST` only if you deliberately want it network-reachable.
2. **API key.** Set `PROXY_API_KEY` to require
   `Authorization: Bearer <key>` on every route. Missing or wrong credentials
   return `401` with an Ollama-style `{"error": "..."}` body. Leave it unset
   only when the proxy is reachable exclusively over loopback.

Because the proxy is exposed through a Cloudflare tunnel, **put Cloudflare
Access in front of the tunnel** (Zero Trust → Access → Applications) and
require an identity policy for the hostname. The bearer token is a second
layer, not a substitute for Access: it protects against someone who finds the
tunnel URL, but Access gives you per-user auth, audit logs, and revocation.

Run `cloudflared` on the same host as the proxy, e.g.:

```bash
cloudflared tunnel --url http://127.0.0.1:11434
```

## Tests

Tests use a fake upstream (`httpx.MockTransport`); no GPU or vLLM is needed.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Covered: non-streaming error translation, mid-stream failure ending with a
`done` line, upstream 500 on `stream=true`, the `/api/generate` system prompt
mapping, dropped-option warnings, and auth enabled/disabled.
