#!/usr/bin/env bash
# Trimmed close-out (DESIGN.md change log 2026-11-14): only what resolves F2. F3, F4, F5 and the second profile are
# dropped; their expected outcomes would not change any conclusion.
cd -- "$(dirname -- "$0")/.."
log() { echo "$(date '+%F %T') $*" >> runs/sweep.log; }
uv run python scripts/sweep.py baseline-ctl F2-chunk8192 >> runs/sweep.stdout 2>&1
docker rm -f sl-vllm >/dev/null 2>&1
log "night6 finished"
