#!/usr/bin/env bash
# Remaining work after the 16:40 pause (DESIGN.md change log). Run only with every GPU-using Windows application
# closed: the runner sees video decode, not browser or Discord rendering.
cd -- "$(dirname -- "$0")/.."
log() { echo "$(date '+%F %T') $*" >> runs/sweep.log; }
uv run python scripts/sweep.py F2-chunk8192 baseline-ctl F5-1.5B F3-off F4-kv16k F4-kv64k >> runs/sweep.stdout 2>&1
docker rm -f sl-vllm >/dev/null 2>&1
mkdir -p runs/_profile_r1 && mv runs/_profile/traces runs/_profile_r1/ 2>/dev/null
uv run python scripts/profile_decode.py run > runs/profile2.stdout 2>&1
log "profile2 exit $?"
docker rm -f sl-vllm >/dev/null 2>&1
log "night5 finished"
