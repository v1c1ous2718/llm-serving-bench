"""Kernel-level floor for one batch-1 decode step of Qwen2.5-0.5B, without vLLM (runs inside the pinned image).

  sg docker -c "docker run --rm --gpus all -v $PWD/scripts:/s --entrypoint python3 vllm/vllm-openai:v0.32.0 /s/microbench.py"

1. Achievable DRAM bandwidth: device-to-device copy of 1 GiB (bytes read + written).
2. The step's weight GEMVs (BF16, batch 1, torch.matmul -> cuBLAS), 24 layers + lm_head, replayed as one CUDA
   graph so launch overhead is excluded. Weight bytes / graph time = bandwidth the GEMVs actually reach.
3. The same GEMVs launched eagerly, one kernel at a time: the gap to (2) is per-kernel launch cost.
"""
import json

import torch

dev = "cuda"
H, I, L, V = 896, 4864, 24, 151936
QKV = H + 2 * 2 * 64  # 14 query heads + 2 KV heads (K and V) of dim 64


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}

a = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
b = torch.empty_like(a)
ms = timeit(lambda: b.copy_(a), 20)
res["copy_GBps"] = 2 * (1 << 30) / ms / 1e6
del a, b

shapes = [(QKV, H), (H, H), (2 * I, H), (H, I)] * L + [(V, H)]
Ws = [torch.randn(o, i, dtype=torch.bfloat16, device=dev) for o, i in shapes]
x = {i: torch.randn(1, i, dtype=torch.bfloat16, device=dev) for i in (H, I)}
wbytes = sum(w.numel() * 2 for w in Ws)
res["weight_GB"] = wbytes / 1e9


def step():
    for w in Ws:
        torch.matmul(x[w.shape[1]], w.t())


res["eager_ms"] = timeit(step)
g = torch.cuda.CUDAGraph()
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    step()
torch.cuda.current_stream().wait_stream(st)
with torch.cuda.graph(g):
    step()
res["graph_ms"] = timeit(g.replay)
res["graph_GBps"] = wbytes / res["graph_ms"] / 1e6
res["lm_head_ms"] = timeit(lambda: torch.matmul(x[H], Ws[-1].t()))
per = {}
for name, (o, i) in {"qkv": (QKV, H), "o_proj": (H, H), "gate_up": (2 * I, H), "down": (H, I),
                     "lm_head": (V, H)}.items():
    # Cycle through copies totalling > 64 MB so the 3 MB L2 cache cannot serve the weights.
    k = max(1, int(64e6 // (o * i * 2)) + 1)
    ws = [torch.randn(o, i, dtype=torch.bfloat16, device=dev) for _ in range(k)]
    xv = torch.randn(i, dtype=torch.bfloat16, device=dev)
    mb = o * i * 2 / 1e6
    t_mm = timeit(lambda: [torch.matmul(xv[None], w.t()) for w in ws], 20) / k
    t_mv = timeit(lambda: [torch.mv(w, xv) for w in ws], 20) / k
    del ws
    per[name] = {"MB": mb, "matmul_us": t_mm * 1e3, "matmul_GBps": mb / t_mm / 1e3,
                 "mv_us": t_mv * 1e3, "mv_GBps": mb / t_mv / 1e3}
res["per_shape"] = per
res["bound_at_copy_bw_ms"] = wbytes / (res["copy_GBps"] * 1e6)
print(json.dumps(res, indent=1))
