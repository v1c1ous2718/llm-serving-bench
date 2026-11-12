#!/usr/bin/env bash
# Start the pinned vLLM server for one configuration (DESIGN.md "Setup").
#   scripts/serve.sh <run-name> [extra vLLM args...]
#   scripts/serve.sh baseline
#   scripts/serve.sh F1-seqs32 --max-num-seqs 32
# Model defaults to Qwen2.5-0.5B-Instruct; override with MODEL=Qwen/Qwen2.5-1.5B-Instruct.
# The full command, image digest and startup log are saved under runs/_server/<run-name>/.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

IMAGE=vllm/vllm-openai:v0.32.0
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
name=${1:?run name required}; shift
dir=runs/_server/$name
mkdir -p "$dir"

docker rm -f sl-vllm >/dev/null 2>&1 || true
# Later arguments override earlier ones, so a factor can still change the memory fraction.
args=(--model "$MODEL" --served-model-name qwen --dtype bfloat16 --max-model-len 8192
      --gpu-memory-utilization 0.75 "$@")
{
  echo "date: $(date -Iseconds)"
  echo "image: $IMAGE"
  echo "digest: $(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
  echo "args: ${args[*]}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$dir/server.txt"

docker run -d --name sl-vllm --gpus all --ipc=host -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" "${args[@]}" >/dev/null

# Wait for readiness, then keep the startup log (KV-cache block count, pinned-memory warning).
for _ in $(seq 1 180); do
  if curl -sf localhost:8000/health >/dev/null; then break; fi
  sleep 2
done
docker logs sl-vllm > "$dir/startup.log" 2>&1
curl -sf localhost:8000/health >/dev/null || { echo "server not ready; see $dir/startup.log"; exit 1; }
grep -iE "gpu blocks|kv cache|pin" "$dir/startup.log" || true
echo "ready: $name"
