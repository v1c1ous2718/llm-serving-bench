"""Kernel time per decode step for one variant (CUDA-graph replays under the torch profiler).
  kernels/run.sh profile_step.py E3 [batch]
Prints kernel time per step grouped by kernel name, GPU busy time, and the replay wall time.
"""
import collections
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from engine import Engine, find_model

v = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
kw = {"E1": {}, "E2": {"gemv": True}, "E3": {"gemv": True, "attn": True}, "E4": {"fused": True}}[v]
eng = Engine(find_model(), max_batch=8, max_ctx=1024, **kw)
prompt = torch.randint(0, 150000, (B, 512), device="cuda")
eng.prefill(prompt)
eng.capture(B, pos0=512)
for _ in range(20):
    eng.graph.replay()
torch.cuda.synchronize()
N = 50
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        eng.graph.replay()
    torch.cuda.synchronize()
by = collections.defaultdict(lambda: [0.0, 0])
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        by[e.name[:80]][0] += e.device_time / N
        by[e.name[:80]][1] += 1 / N
total = sum(t for t, _ in by.values())
print(f"{v} batch {B}: kernel time per step {total / 1e3:.3f} ms")
for name, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:18]:
    print(f"{t:9.1f} us  {c:5.1f}x  {name}")
