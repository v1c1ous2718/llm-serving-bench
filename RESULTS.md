# Results

All numbers come from `runs/`; `sl/summarize.py` recomputes them. Server: `vllm/vllm-openai:v0.32.0`,
Qwen2.5-0.5B-Instruct BF16, `--gpu-memory-utilization 0.85` (KV cache 738,752 tokens = 8.45 GiB / 12 KiB per
token, matching the bound computed in DESIGN.md).

## F0: roofline check (W1, closed loop, seed 0) — 2026-11-11

| In-flight requests | Output tok/s | TPOT p50 | TPOT p95 | TTFT p50 |
|---:|---:|---:|---:|---:|
| 1 | 215 | 4.58 ms | 4.67 ms | 18 ms |
| 2 | 390 | 5.00 ms | 5.16 ms | 17 ms |
| 4 | 758 | 5.05 ms | 5.26 ms | 17 ms |
| 8 | 1,334 | 5.39 ms | 5.47 ms | 18 ms |
| 16 | 2,016 | 6.04 ms | 6.24 ms | 22 ms |
| 32 | 2,535 | 7.90 ms | 8.71 ms | 31 ms |
| 64 | 3,500 | 11.94 ms | 13.62 ms | 62 ms |
| 128 | 3,881 | 24.78 ms | 29.74 ms | 114 ms |
| 256 | 4,046 | 54.24 ms | 65.13 ms | 241 ms |

Throughput at 1–8 in-flight requests includes ramp-up and drain, so it understates the steady state.

**Prediction 1 (batch-1 TPOT within 1.5× of 2.8 ms): refuted.** 4.58 ms is 1.64× the bound, an effective
0.99 GB / 4.58 ms ≈ 216 GB/s, 60% of the 360 GB/s spec. Not yet explained; candidates to test with
Nsight Systems: per-step CPU work (scheduling, sampling, detokenization) between GPU steps, and small
GEMVs that do not reach full bandwidth.

**Prediction 2 (tokens/s roughly linear in batch up to a knee): supported.** From 1 to 8 in-flight requests
TPOT rises only 18% (4.58 → 5.39 ms) while throughput grows 6.2×, because each step's weight read is shared.
The knee lies between 16 and 64; past 128 throughput is flat at ~4,000 tok/s and TPOT grows in proportion to
batch size, so the step is no longer dominated by the weight read.
Not yet explained either: at 256 sequences a step reads ~1 GB of weights plus roughly 256 × ~300 tokens ×
12 KiB ≈ 0.9 GB of KV cache, so KV reads may now cost as much as the weights. To be checked with a
profile before it is stated as a finding.

Derived for the sweeps: W1 closed-loop capacity C ≈ 1,024 requests / 55.2 s ≈ 18.5 req/s. The highest open-loop
point (1.25 C ≈ 23 req/s) is well under the client ceiling (100 req/s), and 3 C ≈ 56 req/s also is.
Under the TPOT SLO (p95 ≤ 50 ms), 128 in-flight requests pass and 256 do not.
