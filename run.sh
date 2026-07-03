#!/bin/bash
# Launches: gemma4 (8001), deepseek-ocr (8002), nomic-embed-text (8003)
# all bound to 127.0.0.1 only (not exposed), then the Ollama-compatible
# proxy on 0.0.0.0:11434, which is the ONLY port Cloudflare needs to hit.
set -e

mkdir -p logs

echo "[1/4] Starting gemma4 (google/gemma-4-E4B-it) on :8001 ..."
nohup vllm serve google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  > logs/gemma4.log 2>&1 &

echo "[2/4] Starting deepseek-ocr (deepseek-ai/DeepSeek-OCR) on :8002 ..."
nohup vllm serve deepseek-ai/DeepSeek-OCR \
  --host 127.0.0.1 --port 8002 \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  > logs/deepseek-ocr.log 2>&1 &

echo "[3/4] Starting nomic-embed-text on :8003 ..."
nohup vllm serve nomic-ai/nomic-embed-text-v1.5 \
  --host 127.0.0.1 --port 8003 \
  --task embed \
  --trust-remote-code \
  --gpu-memory-utilization 0.15 \
  > logs/nomic-embed.log 2>&1 &

echo "Waiting for backends to come up (this can take a few minutes on first run, they're downloading weights)..."
for port in 8001 8002 8003; do
  until curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
    sleep 5
    echo "  ...still waiting on :${port}"
  done
  echo "  :${port} is up"
done

echo "[4/4] Starting Ollama-compatible proxy on :11434 ..."
nohup python proxy.py > logs/proxy.log 2>&1 &

echo ""
echo "All up. Point Cloudflare / your app at port 11434, exactly like before."
echo "Logs: logs/gemma4.log, logs/deepseek-ocr.log, logs/nomic-embed.log, logs/proxy.log"
echo "Tail everything with: tail -f logs/*.log"