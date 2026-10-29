#!/bin/bash
# Launches: gemma4 (8001), deepseek-ocr (8002), nomic-embed-text (8003)
# all bound to 127.0.0.1 only (not exposed), then the Ollama-compatible
# proxy on 127.0.0.1:11434, which is the ONLY port Cloudflare needs to hit.
#
# Each vLLM server runs under a supervisor that restarts it with exponential
# backoff if it crashes. Readiness is bounded by READY_TIMEOUT (seconds,
# default 900); if a server is not ready in time, run.sh exits non-zero and
# prints the tail of that server's log.
#
# Environment:
#   READY_TIMEOUT           seconds to wait per backend (default 900)
#   RESTART_BACKOFF_START   first restart delay in seconds (default 5)
#   RESTART_BACKOFF_MAX     maximum restart delay in seconds (default 300)
#   HOST                    proxy bind address (default 127.0.0.1)
#   PROXY_API_KEY           if set, require "Authorization: Bearer <key>"
set -e

mkdir -p logs

READY_TIMEOUT="${READY_TIMEOUT:-900}"
RESTART_BACKOFF_START="${RESTART_BACKOFF_START:-5}"
RESTART_BACKOFF_MAX="${RESTART_BACKOFF_MAX:-300}"
# Treat an empty HOST as unset, matching proxy.py: never bind every interface.
PROXY_HOST="${HOST:-127.0.0.1}"

# PIDs of the background supervisor bash processes, so we can shut them down
# (and their vLLM children) on any exit path.
supervisor_pids=()

cleanup() {
  local code=$?
  # Stop every supervisor and its vLLM children so a failed or interrupted
  # run can't leave supervisors restarting vLLM behind it.
  local pid child
  for pid in "${supervisor_pids[@]}"; do
    [ -n "$pid" ] || continue
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
      kill "$child" 2>/dev/null || true
    done
    kill "$pid" 2>/dev/null || true
  done
  exit "$code"
}
trap cleanup EXIT INT TERM

# Keep one vLLM server alive: restart it with capped exponential backoff.
# The backoff resets after a run that stayed up for at least 60 seconds.
supervise() {
  local name="$1" log="$2"
  shift 2
  local backoff="$RESTART_BACKOFF_START"
  while true; do
    local started code=0
    started=$(date +%s)
    "$@" >> "$log" 2>&1 || code=$?
    local uptime=$(( $(date +%s) - started ))
    if [ "$uptime" -ge 60 ]; then
      backoff="$RESTART_BACKOFF_START"
    fi
    echo "[supervisor] $(date -Is) ${name} exited (code ${code}) after ${uptime}s; restarting in ${backoff}s" >> "$log"
    sleep "$backoff"
    if [ "$uptime" -lt 60 ] && [ "$backoff" -lt "$RESTART_BACKOFF_MAX" ]; then
      backoff=$(( backoff * 2 ))
      [ "$backoff" -gt "$RESTART_BACKOFF_MAX" ] && backoff="$RESTART_BACKOFF_MAX"
    fi
  done
}

export -f supervise
export RESTART_BACKOFF_START RESTART_BACKOFF_MAX

start_supervised() {
  # start_supervised <name> <logfile> <command...>
  local name="$1" log="$2"
  shift 2
  nohup bash -c 'supervise "$@"' _ "$name" "$log" "$@" > /dev/null 2>&1 &
  supervisor_pids+=("$!")
}

echo "[1/4] Starting gemma4 (google/gemma-4-E4B-it) on :8001 ..."
start_supervised gemma4 logs/gemma4.log \
  vllm serve google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 32768 \
  --max-num-seqs 16

echo "[2/4] Starting deepseek-ocr (deepseek-ai/DeepSeek-OCR) on :8002 ..."
start_supervised deepseek-ocr logs/deepseek-ocr.log \
  vllm serve deepseek-ai/DeepSeek-OCR \
  --host 127.0.0.1 --port 8002 \
  --enforce-eager \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16

echo "[3/4] Starting nomic-embed-text on :8003 ..."
start_supervised nomic-embed-text logs/nomic-embed.log \
  vllm serve nomic-ai/nomic-embed-text-v1.5 \
  --host 127.0.0.1 --port 8003 \
  --runner pooling \
  --trust-remote-code \
  --gpu-memory-utilization 0.15

log_for_port() {
  case "$1" in
    8001) echo logs/gemma4.log ;;
    8002) echo logs/deepseek-ocr.log ;;
    8003) echo logs/nomic-embed.log ;;
    *) echo "" ;;
  esac
}

echo "Waiting for backends to come up (this can take a few minutes on first run, they're downloading weights)..."
echo "Readiness timeout: ${READY_TIMEOUT}s per backend."
for port in 8001 8002 8003; do
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  until curl -sf --max-time 5 "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log=$(log_for_port "$port")
      echo "ERROR: backend on :${port} was not ready within ${READY_TIMEOUT}s." >&2
      echo "--- tail of ${log} ---" >&2
      tail -n 50 "$log" >&2 || true
      echo "--- end of ${log} ---" >&2
      echo "Increase READY_TIMEOUT if the model is still downloading weights." >&2
      exit 1
    fi
    sleep 5
    echo "  ...still waiting on :${port}"
  done
  echo "  :${port} is up"
done

echo ""
echo "All up. Point Cloudflare / your app at port 11434, exactly like before."
echo "The proxy listens on ${PROXY_HOST}:11434; keep cloudflared on this host."
echo "Logs: logs/gemma4.log, logs/deepseek-ocr.log, logs/nomic-embed.log, logs/proxy.log"
echo "Tail everything with: tail -f logs/*.log"
echo "[4/4] Starting Ollama-compatible proxy on ${PROXY_HOST}:11434 ..."
# Run in the foreground so the EXIT trap shuts down the vLLM supervisors when
# the proxy stops (Ctrl-C, crash, or `exit`).
python proxy.py > logs/proxy.log 2>&1
