#!/usr/bin/env bash
# After the sweep: attention-backend comparison at large batch (mixed prefill+decode steps), then the
# decode profile again with clients long enough for c = 64 and 256.
cd -- "$(dirname -- "$0")/.."
while pgrep -f "python scripts/sweep[.]py" >/dev/null; do sleep 60; done
for b in TRITON_ATTN FLASHINFER; do
  F0_CONC=8,64,128,256 uv run python scripts/f0.py "F0-$b" --attention-backend "$b" > "runs/f0-$b.stdout" 2>&1
  echo "$(date '+%F %T') F0-$b exit $?" >> runs/sweep.log
done
mkdir -p runs/_profile_r1 && mv runs/_profile/traces runs/_profile_r1/ 2>/dev/null
uv run python scripts/profile_decode.py run > runs/profile2.stdout 2>&1
echo "$(date '+%F %T') profile2 exit $?" >> runs/sweep.log
docker rm -f sl-vllm >/dev/null 2>&1
echo "$(date '+%F %T') night2 finished" >> runs/sweep.log
