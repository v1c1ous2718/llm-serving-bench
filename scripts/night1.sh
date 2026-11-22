#!/usr/bin/env bash
# F0 rerun, decode profile, kernel microbenchmark; one after another, each logged.
cd -- "$(dirname -- "$0")/.."
uv run python scripts/f0.py F0-r2 > runs/f0-r2.stdout 2>&1; echo "f0 exit $?"
uv run python scripts/profile_decode.py run > runs/profile.stdout 2>&1; echo "profile exit $?"
docker rm -f sl-vllm >/dev/null 2>&1
mkdir -p runs/_profile
docker run --rm --gpus all -v "$PWD/scripts:/s:ro" --entrypoint python3 vllm/vllm-openai:v0.32.0 /s/microbench.py \
  > runs/_profile/microbench.json 2> runs/_profile/microbench.err; echo "microbench exit $?"
