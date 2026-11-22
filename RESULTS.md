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

## F0 rerun on a quiet GPU (`runs/F0-r2`) — 2026-11-13

Same server and points as F0, all Windows applications closed (card at 741 MiB, decoder 0% before start).
Throughput is now the **steady state**: output chunks inside the window in which all c slots are busy
(`scripts/f0.py`). The first F0 table included ramp-up and drain unevenly, so its large-batch tokens/s were
too low; recomputed with the steady-state rule, the first run's raw data agree with the rerun within 1.5%.

| In-flight | Steady tok/s | TPOT p50 | TPOT p95 | TTFT p50 | (first run, same rule: tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 216 | 4.57 ms | 4.59 ms | 17 ms | 215 |
| 2 | 392 | 5.04 ms | 5.12 ms | 16 ms | 393 |
| 4 | 777 | 5.09 ms | 5.16 ms | 17 ms | 779 |
| 8 | 1,457 | 5.43 ms | 5.49 ms | 18 ms | 1,469 |
| 16 | 2,634 | 6.01 ms | 6.16 ms | 22 ms | 2,619 |
| 32 | 3,995 | 8.01 ms | 8.77 ms | 30 ms | 4,000 |
| 64 | **5,392** | 12.07 ms | 13.44 ms | 53 ms | 5,329 |
| 128 | 5,090 | 25.01 ms | 28.95 ms | 108 ms | 5,063 |
| 256 | 4,628 | 55.78 ms | 64.56 ms | 235 ms | 4,656 |

Batch-1 drift over the session: 4.57 → 4.57 ms (−0.1%), valid. Background GPU load did not cause the 4.58 ms.
**Correction to F0:** throughput does not flatten past 128; it peaks at 64 and falls.

## Why batch-1 decode reaches only 60% of the bandwidth bound

Sources: vLLM torch profiler, 100 engine steps at c = 1 (`scripts/profile_decode.py`, `runs/_profile_r1/`);
a kernel microbenchmark without vLLM (`scripts/microbench.py`, `runs/_profile/microbench.json`; its per-shape
`*_GBps` fields are 1000× too small, a unit slip, and the per-shape eager timings of the 2 MB matrices are
launch-bound, so per-projection numbers below come from the vLLM trace). Recompute: `scripts/decode_breakdown.py`.

- **The GPU is not idle.** Kernels cover 95.6% of each 4.64 ms step (profiled). CPU-side work between steps
  (scheduling, sampling, detokenization) costs at most ~0.2 ms. Hypothesis "per-step CPU work" is refuted.
- **Achievable DRAM bandwidth is 314 GB/s**, not 360 (1 GiB device copy). At 314 GB/s the weights take 3.15 ms.
- **The weight GEMMs reach 246 GB/s on average** inside layers; cuBLAS picks a tensor-core GEMM tile for M = 1
  on the small projections:

  | Projection (×24) | MB per layer | Time per step | Bandwidth | Kernel |
  |---|---:|---:|---:|---|
  | qkv | 2.1 | 0.24 ms | 203 GB/s | cutlass wmma 16×16 |
  | o_proj | 1.6 | 0.28 ms | 140 GB/s | cutlass tensorop 64×64 |
  | gate_up | 17.4 | 1.43 ms | 293 GB/s | cuBLAS gemv |
  | down | 8.7 | 0.96 ms | 219 GB/s | cutlass wmma, split-K 3 |
  | lm_head (once) | 272 | 0.88 ms | 308 GB/s | cuBLAS gemv |

  The same GEMVs alone, replayed in a CUDA graph without vLLM: 3.54 ms (279 GB/s).
- **Work the bound did not count:** attention 0.43 ms and norms / RoPE / KV-cache write / activation / sampling
  0.26 ms per step.

Decomposition of 4.57 ms against the 2.8 ms bound (1.64×): spec → achievable DRAM bandwidth 1.15×
(2.75 → 3.15 ms); small-matrix GEMM efficiency 1.12× (→ 3.54 ms); non-weight kernels and step overhead 1.29×
(→ 4.57 ms). 1.15 × 1.12 × 1.29 = 1.66.

## Why throughput falls past 64 in-flight requests

The F0 hypothesis "at 256 the KV reads cost as much as the weights" is **refuted**: in a decode-only step at 256,
attention is 5.3 ms of a 15.5 ms GPU span. The cause is steps that mix one or more prefills into the decode batch.

- Profiled at c = 256: decode-only steps span 15.5 ms; steps with any prefill (79–1,151 prompt tokens) span
  **241 ms**, of which `flash_fwd_splitkv_kernel` is 212 ms. The profiler inflates this (see next point), but the
  kernel choice for mixed batches is the suspect.
- Without the profiler, inter-token gaps in F0-r2 are bimodal: at 256, decode-only gaps are ~20.5 ms and 37% of
  gaps are more than twice that; those slow gaps hold **74%** of all decode time (40% at 64, 60% at 128). At 20.5 ms
  per step, 256 sequences would give ~12,500 tok/s, against 4,628 measured.
- Closed-loop W1 at high concurrency admits a new request about every step, so most steps are mixed.
  Next check (queued after the sweep, `scripts/night2.sh`): the same points with `--attention-backend TRITON_ATTN`
  and `FLASHINFER`. If the penalty is specific to FLASH_ATTN's split-KV path, the other backends lose it.

## Baseline sweep (vLLM defaults, `--gpu-memory-utilization 0.75`) — 2026-11-13

3 seeds × 600 requests per point, first 50 discarded, p95 pooled over seeds (`scripts/analyze.py`). Session drift
4.54 → 4.54 ms (−0.1%), valid. No preemptions at any point. Rates are multiples of each workload's closed-loop
capacity C at 256 in flight: W1 16.42, W2 4.45, Wmix 9.91, W3 28.99 req/s.

| Workload | Goodput | At the first failing rate | Limiting term |
|---|---|---|---|
| W1 chat | **20.5 req/s** (1.25 C) | 1.5 C: TTFT p95 49 s, TPOT p95 215 ms | both, collapse |
| W2 long prompt | **4.0 req/s** (0.9 C) | 1.0 C: TPOT p95 51.7 ms | TPOT |
| Wmix | **12.4 req/s** (1.25 C) | 1.5 C: TTFT p95 46 s, TPOT p95 266 ms | both, collapse |
| W3 shared prefix | **20.3 req/s** (0.7 C) | 0.8 C: TTFT p95 8.3 s, TPOT p95 178 ms | both, collapse |

- **C understates what open loop sustains for W1 and Wmix**: at 1.25 C both still meet the SLO with room
  (W1 TPOT p95 17.9 ms). Closed loop at 256 in flight runs past the throughput peak at 64 (F0-r2), so C is a low
  estimate whenever an open-loop rate keeps fewer requests in flight. Higher multiples were added (DESIGN.md).
- **Past goodput the system collapses, it does not degrade** (W1, Wmix, W3): one rate step takes TPOT p95 from
  11–18 ms to 200+ ms and TTFT p95 to tens of seconds. 1.5–2.0 C ran in `baseline-hi` (drift checked, same args).
- **W2 is TPOT-bound**: its 2,000-token prompts are prefilled in 2,048-token chunks inside decode steps, so every
  admitted prompt stretches the steps of all running decodes; TPOT p95 grows from 7 ms at 0.1 C to 52 ms at 1.0 C
  while TTFT p95 stays under 1 s.
- **W3 collapses instead of degrading**: server-side running requests at 0.7 C climb from ~40 to 129 during the
  25-s point without settling; at 0.8 C they hit the 256 cap within ~15 s and requests start to wait (up to 143).
  Same mechanism as F0's falling throughput: more requests in flight make prefill-mixed steps slower, which
  lowers the service rate and admits still more requests. Every W3 arrival carries a (100-token, prefix-cached)
  prefill, so mixed steps are frequent. Caveat: 600-request points are short; with the upward drift at 0.7 C,
  a longer run could fail there too, so 20.3 req/s is an upper estimate.
- One W2 request at 1.0 C (seed 2) failed with `ServerDisconnectedError` 0.9 ms after sending: the client reused a
  pooled keep-alive connection the server had just closed. Client-side, not load-related; reported, not counted
  in the SLO (DESIGN.md defines goodput by the two latency terms).

## Attention backend comparison (F0 points, W1 closed loop, 0.85) — 2026-11-13

Test of the hypothesis above: if the penalty for prefill-mixed steps lives in FLASH_ATTN's split-KV path, other
backends should not show it. Same server otherwise; `--attention-backend` only. Steady-state tokens/s:

| In-flight | FLASH_ATTN (default, F0-r2) | TRITON_ATTN | FLASHINFER |
|---:|---:|---:|---:|
| 1 (TPOT p50) | 4.57 ms | **4.27 ms** | 4.36 ms |
| 8 | 1,457 | 1,498 | 1,490 |
| 64 | 5,392 | 6,648 | 6,846 |
| 128 | 5,090 | 8,054 | 8,106 |
| 256 | 4,628 | **8,868** (TPOT p50 29.6 ms) | 8,894 (TPOT p50 29.5 ms; rerun) |

- **Hypothesis supported.** With TRITON_ATTN throughput keeps rising to 256 (1.92× FLASH_ATTN there; TPOT p50
  55.8 → 29.6 ms). Decode-only steps are unchanged (inter-token gap p50 21.9 vs 21.5 ms at 256); the share of
  decode time in gaps over 41 ms (2× a decode-only step) drops from 74% to 23%, and gap p99 from 154 to 81 ms.
- Batch 1 also gains 0.3 ms per token (4.57 → 4.27 ms): FLASH_ATTN's split-KV kernel plus its combine kernel
  cost 16 µs per layer at batch 1 (0.38 ms per step, `runs/_profile_r1`).
- FLASHINFER matches TRITON_ATTN at every point. Its first session's 256 point returned non-200 responses for
  some requests and the client's error path (missing `end` timestamp, now fixed in `sl/client.py`) lost the file;
  the server log was overwritten by the next session. The rerun (`runs/F0-FLASHINFER-r2`, 07:25, server log
  kept) gave 0 errors in 1,024 requests; the first failure did not reproduce. The FLASHINFER column above is
  c = 1–128 from the first session and c = 256 from the rerun (the rerun's c = 1–128: 227 / 1,509 / 6,921 / 8,157).
- Not yet explained: why vLLM v0.32.0 picks FLASH_ATTN on this card, and what in its split-KV path makes mixed
  batches slow (a per-kernel profile of TRITON_ATTN's mixed steps is the next step).

## F6: attention backend in the open-loop sweep — 2026-11-13

`--attention-backend TRITON_ATTN`, otherwise the baseline; W1 and W3 at the baseline's absolute rates, 3 seeds.
Drift 4.27 → 4.25 ms (−0.4%), valid. No preemptions, no errors. Prediction (DESIGN.md, written before the run):
W3 gains the most, W1 less. **Supported**, with both gains larger than expected:

| Workload | Goodput, FLASH_ATTN | Goodput, TRITON_ATTN | At the highest tested rate (2.0 C), TRITON_ATTN |
|---|---|---|---|
| W1 | 20.5 req/s | **≥ 32.8 req/s** (≥ 1.6×) | TTFT p95 0.064 s, TPOT p95 13.2 ms |
| W3 | 20.3 req/s | **≥ 58.0 req/s** (≥ 2.9×) | TTFT p95 0.066 s, TPOT p95 13.2 ms |

- Neither workload fails at 2.0 C, so both are lower bounds; the next session needs 2.5–4 C (W3 at 4 C = 116 req/s
  exceeds the client ceiling of 100 req/s, so it would need two client processes, CHECKS.md).
- The collapse is gone: at W3 0.8 C, where FLASH_ATTN reached TTFT p95 8.3 s, TRITON_ATTN gives 0.039 s, and TPOT
  p95 grows smoothly from 5.3 to 13.2 ms over a 20× rate range.

## F1: `max_num_seqs` (W1) — in progress

| Value | Goodput | Limiting term | Notes |
|---|---|---|---|
| 8 | **1.6 req/s** (0.1 C; fails at 0.25 C, 4.1 req/s) | TTFT | TPOT p95 stays 5.4–5.6 ms at every rate; closed-loop capacity 4.92 req/s. Drift −0.2%, valid. |
| 32 | **11.5 req/s** (0.7 C; fails at 0.8 C, 13.1 req/s) | TTFT | TPOT p95 ≤ 8.3 ms everywhere; closed-loop capacity 11.61 req/s. Drift +1.1%, valid. One point (0.1 C seed 2) had a Windows video decoding for 115 of 332 s (runner checks only before a point); its TPOT p95 6.6 vs ~5.5 ms in the other seeds. It cannot change goodput (0.1 C passes either way) and is kept, flagged by `analyze.py`. |
| 128 | **20.5 req/s** (1.25 C; fails at 1.5 C) | TTFT | At 1.5 C: TTFT p95 1.8 s, TPOT p95 24 ms, against 49 s / 215 ms with 256. Drift −1.4%, valid. |
| 256 (baseline) | 20.5 req/s | both, collapse | |

Prediction (DESIGN.md: small values keep TPOT low but saturate at a lower rate) **supported** for 8: with only 8
slots TPOT never moves, and every request past the 8th waits in the queue, so TTFT p95 reaches 1.8 s at 4.1 req/s.
For the middle-peak part: 128 ties 256 at this rate resolution (20.5 req/s), so the peak is not resolved. The
difference is past goodput: capped at 128, the server queues instead of collapsing (TTFT p95 1.8 s and TPOT p95
24 ms at 1.5 C, against 49 s and 215 ms at 256), because the cap keeps the batch from reaching the sizes where
prefill-mixed steps get slow enough to feed the collapse (F0 analysis). Under overload, 128 degrades gracefully.

## F2: chunked prefill (Wmix) — in progress

| Value | Goodput | Notes |
|---|---|---|
| on, 2,048 (baseline) | 12.4 req/s (1.25 C) | collapse at 1.5 C: TTFT p95 46 s, TPOT p95 266 ms |
| **off** | **≥ 19.8 req/s** (≥ 2.0 C, not bracketed) | TPOT p95 32.7 ms, TTFT p95 0.14 s at 2.0 C. Drift −0.2%, valid. |
| on, 512 | **≥ 19.8 req/s** (≥ 2.0 C, not bracketed) | TPOT p95 27.7 ms, TTFT p95 0.12 s at 2.0 C; same curve as off within noise. Drift +1.6%, valid. Seed-2 points at 0.9–2.0 C ran while Windows apps used the GPU; seeds 0 and 1 alone also pass at 2.0 C (TPOT p95 29.1 / 24.8 ms). |
| on, 8,192 | rerun pending | first session INVALID (drift +14.2%): Windows apps rendered on the GPU from ~15:04 (DESIGN.md). |

- **Open question before interpreting F2:** off and 512 both pass at 2.0 C, while the baseline (2,048) collapsed at
  1.5 C in `baseline-hi` with non-monotonic TTFT (46 / 21 / 27 s at 1.5 / 1.75 / 2.0 C). Either chunks of 2,048
  cause it, or the collapse is metastable and depends on the arrival sample. Control `baseline-ctl` is queued.
  Until it runs, the prediction is neither supported nor refuted.
- vLLM warned at startup: "This model does not officially support disabling chunked prefill. Disabling this
  manually may cause the engine to crash or produce incorrect outputs." The engine did not crash and every
  request returned its full `max_tokens` (`ignore_eos`); output text was not checked, so F2-off is a performance
  result only.
