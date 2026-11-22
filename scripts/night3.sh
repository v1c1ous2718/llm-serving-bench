#!/usr/bin/env bash
# Reordered night run (DESIGN.md change log 2026-11-13): finish the baseline session, then the attention-backend
# comparison, then the remaining sessions with the longest (F4) last, then the second decode profile.
cd -- "$(dirname -- "$0")/.."
log() { echo "$(date '+%F %T') $*" >> runs/sweep.log; }
while [ ! -f runs/baseline/bs1_end/W1/0.jsonl ]; do sleep 30; done
sleep 20  # let sweep.py write sessions.jsonl
for p in $(pgrep -f "python scripts/sweep[.]py"); do kill "$p"; done
sleep 5; for p in $(pgrep -f "sl[.]client") $(pgrep -f "scripts/serve[.]sh"); do kill "$p"; done; sleep 5
docker rm -f sl-vllm >/dev/null 2>&1
# Remove a session the stopped sweep may have just begun, so it restarts cleanly with its own drift points.
[ -d runs/F1-seqs8 ] && [ ! -f runs/F1-seqs8/bs1_end/W1/0.jsonl ] && mv runs/F1-seqs8 "runs/_discarded-F1-seqs8-$(date +%H%M)"
log "night3: baseline complete, sweep paused for the attention-backend comparison"
for b in TRITON_ATTN FLASHINFER; do
  F0_CONC=8,64,128,256 uv run python scripts/f0.py "F0-$b" --attention-backend "$b" > "runs/f0-$b.stdout" 2>&1
  log "F0-$b exit $?"
done
docker rm -f sl-vllm >/dev/null 2>&1
uv run python scripts/sweep.py baseline-hi F1-seqs8 F1-seqs32 F1-seqs128 F2-off F2-chunk512 F2-chunk8192 F5-1.5B F3-off \
  F4-kv16k F4-kv64k >> runs/sweep.stdout 2>&1
mkdir -p runs/_profile_r1 && mv runs/_profile/traces runs/_profile_r1/ 2>/dev/null
uv run python scripts/profile_decode.py run > runs/profile2.stdout 2>&1
log "profile2 exit $?"
docker rm -f sl-vllm >/dev/null 2>&1
log "night3 finished"
# Added 06:25: FLASHINFER's c=256 point returned non-200 responses and the client's error path lost the file.
# Rerun with the fixed client and keep the server log.
F0_CONC=8,64,128,256 uv run python scripts/f0.py F0-FLASHINFER-r2 --attention-backend FLASHINFER > runs/f0-FLASHINFER-r2.stdout 2>&1
log "F0-FLASHINFER-r2 exit $?"
docker logs sl-vllm > runs/_server/F0-FLASHINFER-r2/server-full.log 2>&1
docker rm -f sl-vllm >/dev/null 2>&1
log "night3 finished (after FLASHINFER rerun)"
