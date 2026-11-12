# Design (fixed before running)

Question: on one RTX 3060 (12 GB) serving Qwen2.5-0.5B-Instruct with vLLM, which engine settings move
**goodput** (the highest request rate that still meets a latency SLO), and why? Every finding must be
explained against the hardware bound below, not only reported.

## Hardware bound, computed before any run

| | Qwen2.5-0.5B (BF16) | Qwen2.5-1.5B (BF16) |
|---|---|---|
| Weights read per decode step | ~0.99 GB | ~3.09 GB |
| RTX 3060 memory bandwidth (spec) | 360 GB/s | 360 GB/s |
| **Upper bound, batch-1 decode** | **~360 steps/s (≥ 2.8 ms TPOT)** | **~116 steps/s (≥ 8.6 ms TPOT)** |
| KV cache per token (2 × layers × KV heads × head dim × 2 B) | 12 KiB (24 × 2 × 64) | 28 KiB (28 × 2 × 128) |

Consequences that the experiments must confirm or refute:
- Batch-1 decode is memory-bound; larger batches reuse the same weight read, so output tokens/s should
  grow almost linearly with batch size until compute or KV capacity binds.
- With 0.5B at default memory settings the KV cache holds hundreds of thousands of tokens, so KV capacity
  will **not** bind unless it is restricted on purpose (factor F4).

## Setup
- vLLM OpenAI-compatible server in Docker: image `vllm/vllm-openai:v0.32.0` (released 2026-10-18; chosen over
  v0.33.0, which had no patch release yet). Digest at the first pull (2026-11-11):
  `sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`.
  BF16, `max_model_len` 8192, `--served-model-name qwen`, `--gpu-memory-utilization 0.75` (0.85 for F0; see change log), all other
  settings at that version's defaults (v0.32.0 enables chunked prefill and prefix caching by default).
- Prometheus scrapes vLLM `/metrics` and a small NVML exporter (GPU util, memory, power, SM clock;
  DCGM is not used because its support for GeForce cards under WSL2 is incomplete). Grafana for live
  viewing only; every reported number comes from raw logs.
- Load generator: async Python client on the host, streaming `/v1/completions`, pinned with `taskset` to
  2 of the 12 logical CPUs. It records per request: send time, first-token time, every token time, end time,
  prompt and output token counts.
- Recorded with every run: vLLM tag, driver (currently 621.14), CUDA runtime in the image, Docker version,
  full server arguments, and the KV-cache block count printed by vLLM at startup.

## Workloads
Output length is controlled with `ignore_eos=True` and a sampled `max_tokens`, so every configuration does the
same work. Prompts are sent as token IDs: the GSM8K train questions are tokenized once (Qwen2.5 tokenizer) into
one stream, and each prompt is a slice of the sampled length at a random offset, so prompt lengths are exact.
Lengths are drawn from clipped log-normal distributions (σ = 0.8 for W1, 0.6 for W2) with a fixed seed.

| Name | Prompt tokens (median, clip) | Output tokens (median, clip) | Purpose |
|---|---|---|---|
| W1 chat | 200 (32–1,024) | 200 (16–1,024) | baseline |
| W2 long-prompt | 2,000 (512–6,000) | 100 (16–512) | prefill pressure |
| W-mix | 80% W1 + 20% W2 | | long prefills interfering with decodes |
| W3 shared-prefix | 8 distinct 2,000-token prefixes + 100-token unique suffix | 100 | prefix caching |

Arrivals are Poisson. Each workload is first run closed-loop at maximum concurrency to measure its capacity
C (requests/s). The open-loop sweep then uses rates {0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25} × C.
Each point: 600 requests, the first 50 discarded as warm-up, 3 arrival seeds.

## Metrics
- **TTFT**: first-token time − send time.
- **TPOT**: (end − first token) / (output tokens − 1), per request.
- E2E latency, output tokens/s, completed requests/s.
- p50 / p95 / p99 of TTFT, TPOT and E2E, pooled over the 3 seeds; the spread across seeds is reported.
- **SLO** (fixed now): p95 TTFT ≤ 1.0 s for W1 and W3, ≤ 3.0 s for W2 and W-mix; p95 TPOT ≤ 50 ms
  (≈ 20 tokens/s, faster than reading speed).
- **Goodput**: the highest tested rate at which both SLO terms hold.
- Server side, from `/metrics`: running and waiting request counts, KV-cache usage, preemptions.
  Any point with preemptions is flagged in every plot and table.

## Factors (one at a time from the baseline, like the Food-11 study)
Baseline = vLLM defaults at the pinned tag. If that tag already enables chunked prefill and prefix caching
by default, the factors below switch them off instead of on.

| ID | Factor | Values | Workload | Prediction (written before running) |
|---|---|---|---|---|
| F0 | Roofline check | batch 1, then fixed batch sizes 1–256 | W1, closed loop | batch-1 TPOT within 1.5× of 2.8 ms; tokens/s roughly linear in batch up to a knee |
| F1 | `max_num_seqs` | 8, 32, 128, 256 | W1 | small values keep TPOT low but saturate at a lower rate; goodput peaks in the middle |
| F2 | Chunked prefill, `max_num_batched_tokens` | off; on with 512 / 2,048 / 8,192 | W-mix | chunking lowers p95 TPOT of short requests and raises TTFT of long ones; smaller chunks move both further |
| F3 | Prefix caching | off, on | W3 | on cuts TTFT roughly in proportion to the shared fraction (~95% of prompt tokens); little effect on TPOT |
| F4 | KV-cache capacity | ~16k tokens, ~64k tokens, default | W1 and W2 near capacity | below the working set: preemptions appear and p99 TTFT jumps before throughput drops |
| F5 | Model size | 0.5B, 1.5B | W1 | 1.5B batch-1 TPOT ≈ 3× the 0.5B value (weight bytes ratio); the gap narrows at large batch |

F4 sets capacity with `--num-gpu-blocks-override` if the pinned tag supports it, otherwise with
`--gpu-memory-utilization`; the achieved capacity in tokens is taken from the startup log either way.

## Checks before trusting any number
1. `docker info` reaches the daemon, and `docker run --gpus all <the pinned CUDA image> nvidia-smi` sees the
   GPU inside a container.
2. Client ceiling: against a mock server that streams tokens with no delay, the pinned client must sustain at
   least 3× the highest rate in any sweep. Otherwise the client, not vLLM, is measured.
3. Repeat the F0 batch-1 point at the start and end of each session; drift over 5% invalidates that
   session (thermal throttling or a background GPU load).
4. vLLM under WSL2 may warn that pinned host memory is unavailable. Record whether that warning appears.

## Outputs
- `runs/<factor>/<value>/<workload>/<seed>.jsonl`: one line per request, all timestamps raw.
- `analyze.py` recomputes every table and figure from `runs/` alone.
- Figures: p50/p95 TTFT and TPOT against request rate, one panel per factor, with the SLO lines and the
  hardware bound drawn in; a goodput bar chart per factor.

## Known limits, stated up front
- One consumer GPU under WSL2 virtualization; the client shares the host CPU with vLLM's scheduler
  (mitigated by pinning and checked by test 2).
- Synthetic length distributions and `ignore_eos`; real chat traffic has burstier arrivals and
  data-dependent output lengths.
- Small models: KV capacity binds only where F4 forces it. Results describe this vLLM tag on this card,
  not vLLM in general.
- No multi-GPU behaviour (tensor parallelism, disaggregated prefill) can be measured here.

## Not in this design
A hand-written engine (paged KV cache + continuous batching) and custom Triton kernels come later and will
reuse this load generator and these workloads, so their results are directly comparable.

## Change log
- 2026-11-11, before any result: v0.32.0 defaults to `gpu_memory_utilization` 0.92 (11.04 GiB), but the Windows
  desktop holds ~1 GiB of the 3060, leaving 10.98 GiB free at startup, so the server refused to start.
  The baseline fixes 0.85. The startup log also confirmed that chunked prefill and prefix caching are on by
  default, so F2 and F3 switch them off.
- 2026-11-11, after F0, before any sweep: v0.32.0 resolves `max_num_seqs` 256, `max_num_batched_tokens` 2048,
  block size 16 tokens. So the baseline session already supplies F1 = 256, F2 = on/2048, F3 = on, F4 = default
  and F5 = 0.5B; those values are not rerun. Sweep rates are fixed from the **baseline** capacity C of each
  workload and reused for every factor value, so all values are compared at the same absolute rates; each
  session still records its own closed-loop capacity at 256 in flight. F4 uses `--num-gpu-blocks-override`
  1,000 / 4,000 blocks (16k / 64k tokens). F2 "off" keeps vLLM's own batch budget for unchunked prefill,
  and "on 8,192" gives chunking the same budget. The runner is `scripts/sweep.py`.
- 2026-11-11, first baseline sweep session discarded before any sweep point: batch-1 TPOT read 5.13 ms at the
  start and ~22 ms minutes later, against 4.58 ms in F0. `nvidia-smi` showed the video decoder at 38% and
  12,048 / 12,288 MiB in use: a Windows-side video player shared the card, and with vLLM holding 85% the
  desktop had ~0.2 GiB left. The memory fraction drops to **0.75** for all sweep sessions (KV capacity stays
  above 500k tokens, far beyond F1–F3 working sets, so F0's conclusions are unaffected), and the runner now
  waits for decoder/encoder at 0% and total memory < 11 GiB for 10 s before every point, and logs whole-card
  memory, decoder use, clocks and power at 1 Hz during every point. F0's own `runs/_server/baseline/` record
  was overwritten by this session's start; F0 ran with the arguments in its table header (0.85).
