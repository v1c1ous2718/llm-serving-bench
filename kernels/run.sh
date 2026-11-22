#!/usr/bin/env bash
# Run a script from kernels/ inside the pinned vLLM image (torch 2.13, triton 3.7.1), GPU attached,
# HF cache and a persistent Triton cache mounted.
cd -- "$(dirname -- "$0")"
mkdir -p .triton-cache
docker run --rm --gpus all --ipc=host \
  -v "$PWD:/k" -w /k -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/.triton-cache:/root/.triton" -e HF_HUB_OFFLINE=1 \
  --entrypoint python3 vllm/vllm-openai:v0.32.0 "$@"
rc=$?
# Files written in the container are root-owned; hand them back.
docker run --rm -v "$PWD:/k" --entrypoint chown vllm/vllm-openai:v0.32.0 -R "$(id -u):$(id -g)" /k/results /k/.triton-cache 2>/dev/null || true
exit $rc
