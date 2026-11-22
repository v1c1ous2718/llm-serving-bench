#!/usr/bin/env bash
# Remaining night work, in priority order (DESIGN.md change log 2026-11-13 06:23).
cd -- "$(dirname -- "$0")/.."
log() { echo "$(date '+%F %T') $*" >> runs/sweep.log; }
uv run python scripts/sweep.py baseline-hi >> runs/sweep.stdout 2>&1
docker rm -f sl-vllm >/dev/null 2>&1
rm -rf runs/F0-FLASHINFER-r2 runs/_server/F0-FLASHINFER-r2
F0_CONC=8,64,128,256 uv run python scripts/f0.py F0-FLASHINFER-r2 --attention-backend FLASHINFER > runs/f0-FLASHINFER-r2.stdout 2>&1
log "F0-FLASHINFER-r2 exit $?"
docker logs sl-vllm > runs/_server/F0-FLASHINFER-r2/server-full.log 2>&1
docker rm -f sl-vllm >/dev/null 2>&1
uv run python scripts/sweep.py F6-triton F1-seqs8 F1-seqs32 F1-seqs128 F2-off F2-chunk512 F2-chunk8192 F5-1.5B F3-off \
  F4-kv16k F4-kv64k >> runs/sweep.stdout 2>&1
mkdir -p runs/_profile_r1 && mv runs/_profile/traces runs/_profile_r1/ 2>/dev/null
uv run python scripts/profile_decode.py run > runs/profile2.stdout 2>&1
log "profile2 exit $?"
docker rm -f sl-vllm >/dev/null 2>&1
log "night4 finished"
# Added 15:15: control for baseline-hi's Wmix collapse at 1.5 C (F2-off / F2-chunk512 pass at 2.0 C).
uv run python scripts/sweep.py baseline-ctl >> runs/sweep.stdout 2>&1
docker rm -f sl-vllm >/dev/null 2>&1
log "baseline-ctl done"
