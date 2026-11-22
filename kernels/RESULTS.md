# Results: self-written decode path vs PyTorch defaults and vLLM

RTX 3060 12 GB under WSL2, `vllm/vllm-openai:v0.32.0` image (torch 2.13.0+cu130, triton 3.7.1, transformers 5.19.0),
Qwen2.5-0.5B-Instruct BF16, prompt = first 512 tokens of the model card, greedy, 256 generated tokens.
Raw rows: `results/bench.jsonl` (`bench.py`, `vllm_ref.py`); correctness: `check.py`; kernel profile: `profile_step.py`.
Windows GPU applications closed; video decoder 0% before every run. Measured 2026-11-14.

## Decode time per token

`e2e` = 256 tokens after the prompt / 256 (argmax and input copy included, prefill excluded).
Bandwidth = 0.988 GB of weights per step / step time; floor = 0.988 GB / 314 GB/s measured copy = 3.15 ms.

| ID | Path | Batch 1 e2e | Batch 8 e2e | Batch 1 step, eff. bandwidth |
|---|---|---:|---:|---|
| H0 | HF Transformers eager (default `generate`) | 21.09 ms | — | 19.06 ms, 52 GB/s |
| H1 | HF static cache + `torch.compile` (default mode) | 8.83 ms | — | — |
| E0 | my engine, torch ops, eager | 26.67 ms | — | 24.03 ms |
| E1 | E0 + one CUDA graph per step | 5.62 ms | 6.57 ms | 5.45 ms, 181 GB/s |
| E2 | E1 + my Triton GEMV | 5.30 ms | — | 5.15 ms, 192 GB/s |
| E3 | E2 + my Triton flash-decoding attention | 4.73 ms | — | 4.58 ms, 216 GB/s |
| **E4** | E3 + fused RMSNorm prologue, residual epilogue, RoPE + KV write | **3.56 ms** | **3.82 ms** | **3.43 ms, 288 GB/s** |
| V | vLLM 0.32.0 offline, FLASH_ATTN (default) | 4.28 ms | 5.02 ms | — |
| V | vLLM 0.32.0 offline, TRITON_ATTN | 4.06 ms | 4.73 ms | — |

- **E4 vs PyTorch default (H0): 5.9× per token** (21.09 → 3.56 ms); vs PyTorch compiled (H1): 2.5×.
- **E4 vs vLLM default: 1.20× at batch 1, 1.31× at batch 8**; vs vLLM's faster TRITON_ATTN: 1.14× / 1.24×.
- E4 reaches 288 GB/s, **92% of the measured copy bandwidth**; its step is 1.09× the 3.15 ms floor.
- From batch 1 to 8 E4's step grows 6% (3.43 → 3.65 ms): the weight read is shared; E1 (cuBLAS) grows 17%.

## Where E4's time goes (profile, batch 1, 3.56 ms of kernels incl. profiler overhead)
| Kernel | per step | share |
|---|---:|---:|
| my GEMV (qkv + norm prologue, o + residual, down + residual, lm_head + norm), 73 calls | 2.02 ms | 57% |
| my gate_up GEMV + norm prologue + SiLU·mul, 24 calls | 1.39 ms | 39% |
| my attention split + combine | 0.12 ms | 3% |
| my RoPE + KV-cache write | 0.03 ms | 1% |

Weight GEMVs: 0.987 GB in 3.41 ms = 290 GB/s. The rest of the gap to the floor is GEMV efficiency (290 vs 314 GB/s).

## What each step bought, and the surprise
- E3 profile (before fusing): GEMVs already ran at 289–301 GB/s, but ~950 small torch kernel launches for RMSNorm, RoPE,
  cache writes, dtype casts and residual adds took **1.39 ms per step (29%)**. The prediction had put the remaining
  time in the GEMMs and attention; it was in the glue. E4 (added after this profile, DESIGN.md change log) fuses it
  into the GEMV prologue/epilogue and one RoPE+cache kernel: 7 kernels per layer, glue down to ~0.03 ms.
- My attention: 0.12 ms per step for 24 layers vs 0.43 ms for FLASH_ATTN's split-KV + combine in vLLM, because each
  program reads one KV head once for all 7 query heads sharing it (GQA) and the split count adapts to sequence length.
- At M > 1 the GEMV pads rows to 16 and uses tensor cores (`tl.dot`); the first version reduced in FP32 on CUDA cores
  and lost to cuBLAS at batch 8 (7.71 vs 6.43 ms). Memory traffic is identical; only the arithmetic path changed.

## Predictions (DESIGN.md) vs measured
| ID | Predicted | Measured | |
|---|---|---|---|
| H0 | 20–30 ms | 21.09 | held |
| H1 | 4–6 ms | 8.83 | refuted: `mode="reduce-overhead"` segfaults in this image (exit 139) and inductor skipped CUDA graphs for the static cache ("mutated inputs"); the default mode has no graphs |
| E0 | 10–20 ms | 26.67 | refuted (slower): many more small ops than estimated |
| E1 | 4.0–4.3 ms | 5.62 | refuted: torch attention over a 1,024-key window with a mask; the first version (SDPA + mask + GQA) took 7.44 ms |
| E2 | 3.5–3.8 ms | 5.30 | refuted: GEMVs were fast, the glue was not (see above) |
| E3 | 3.3–3.6 ms | 4.73 | refuted, same reason |
| headline | E3 within 1.15× of 3.15 ms and ≥ 1.25× faster than vLLM | E4: 1.09× of the floor; 1.20× faster than vLLM default at batch 1 (1.31× at batch 8) | floor part held (with E4, not E3); vLLM part narrowly missed at batch 1 |

## Correctness
- Kernels vs torch on the real shapes, M = 1/4/8: max abs error ≤ 0.016 at output magnitudes 2–6 (about one BF16
  ulp); attention ≤ 0.002 for lengths 1–1,000.
- Engine vs HF eager, 8 prompts × 64 greedy tokens (batch 8): E4 matches all 64 tokens on 4–6 of 8 prompts across
  runs, the same range as my pure-torch E0. Prompt 3 diverges at token 0 because HF's own top-2 logits tie there
  (difference 0.0); other divergences start at token 9–60, from BF16 reduction order.
- The model's `generation_config` sets `repetition_penalty: 1.1`, which HF applies even with `do_sample=False`; the
  first comparison used it by mistake and showed early divergence. Reference is plain greedy argmax.

## Limits
Decode step only: no scheduler, continuous batching, paged KV cache, streaming or sampling beyond argmax; batch ≤ 8;
one model on one GPU. vLLM does all of that, so E4 vs V compares the decode step, not a serving system. E1's torch
attention reads a fixed 1,024-key window (so one graph serves every position); E3/E4 read only the live keys.
