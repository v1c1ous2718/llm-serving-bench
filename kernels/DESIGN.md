# Decode kernels: a self-written batch-1 decode path (fixed before running)

## Why this exists
The serving study (`../RESULTS.md`) located where vLLM's batch-1 decode loses time on this RTX 3060: 4.57 ms per
token against a 2.75 ms spec bound (3.15 ms at the measured 314 GB/s). The fix it found, switching vLLM's attention
backend, is a flag, not my work. This part asks: **if I write the decode path myself (PyTorch for the model, my own
Triton kernels for the two bottlenecks the profile named), how close to the DRAM bound do I get, and how does it
compare with PyTorch's default path and with vLLM?**

The two bottlenecks from the profile (`../RESULTS.md`, "Why batch-1 decode reaches only 60%"):
1. Small weight matrices at M = 1: cuBLAS picks tensor-core GEMM tiles; qkv 203, o_proj 140, down 219 GB/s,
   against 293 GB/s for gate_up's GEMV and 314 GB/s for a plain copy.
2. FlashAttention's split-KV decode kernel plus its combine kernel: 16 µs per layer, 0.38 ms per step at short
   context, for 7 query heads × 64 dims over ~200 keys.

## Model and hardware
Qwen2.5-0.5B-Instruct, BF16: hidden 896, 24 layers, 14 query heads / 2 KV heads (GQA 7:1), head dim 64,
MLP 4,864, vocab 151,936, tied embeddings, qkv with bias. Weights read per step: 0.988 GB.
RTX 3060 12 GB, 360 GB/s spec, 314 GB/s measured (1 GiB device copy). Same Docker image as the serving study
(`vllm/vllm-openai:v0.32.0`: torch 2.13.0+cu130, triton 3.7.1, transformers 5.19.0).

## Variants (cumulative), all greedy decode
| ID | Path | What changes |
|---|---|---|
| H0 | HF Transformers, eager, `DynamicCache` (the default `generate` path) | PyTorch default baseline |
| H1 | HF Transformers, static cache + `torch.compile(mode="reduce-overhead")` | PyTorch's own optimized path |
| E0 | My engine: fused weights (qkv, gate_up), static KV cache, torch ops (cuBLAS, SDPA), eager | own model code |
| E1 | E0 captured in one CUDA graph per step | removes launch overhead |
| E2 | E1 + my Triton GEMV for every weight matrix (qkv + bias, o, gate_up fused with SiLU·mul, down, lm_head) | bottleneck 1 |
| E3 | E2 + my Triton GQA decode attention (split-K "flash-decoding": each program reads one KV head once for its 7 query heads, then a combine kernel) | bottleneck 2 |
| V | vLLM v0.32.0 offline `LLM.generate`, defaults (FLASH_ATTN) and TRITON_ATTN | reference |

## Metric
- **Decode step time at batch 1, context 512** (prompt of 512 tokens, step at position 512): median of 200 CUDA-graph
  replays (E1–E3) or 64 consecutive steps with a sync each (H0, H1, E0). Also batch 4 and 8 for E1–E3.
- End-to-end tokens/s for 256 generated tokens after a 512-token prompt, all variants (V included), batch 1.
- Effective bandwidth = 0.988 GB / step time; fraction of 314 GB/s.

## Correctness
- Each Triton kernel against its torch reference on the real weight shapes: max abs error, BF16 tolerance.
- Whole engine against HF eager: greedy tokens for 8 prompts × 64 tokens; report how many match before the first
  divergence, and the max |logit difference| at step 1. BF16 reduction order differs, so some late divergence is
  expected; an early divergence (< 10 tokens) is a bug.

## Predictions (written before any run)
| ID | Predicted ms / token, batch 1, ctx 512 | Reasoning |
|---|---|---|
| H0 | 20–30 | ~15 small kernels per layer launched from Python; launch-bound |
| H1 | 4–6 | CUDA graphs remove launches; same cuBLAS kernels as vLLM |
| E0 | 10–20 | fewer, fused ops than HF, still launch-bound |
| E1 | 4.0–4.3 | vLLM's kernels without its scheduler gaps (vLLM step: 4.57 incl. ~0.2 ms gaps) |
| E2 | 3.5–3.8 | all weight matrices at ≥ 280 GB/s: 0.716 GB in layers / 280 GB/s ≈ 2.56 ms + lm_head 0.87 ms + rest |
| E3 | 3.3–3.6 | attention 0.43 → ~0.15 ms |
| V (FLASH_ATTN) | ~4.4 | server TPOT 4.57 ms minus HTTP/streaming |

Headline prediction: E3 is within 1.15× of the 3.15 ms measured-bandwidth floor and ≥ 1.25× faster than vLLM at
batch 1. If E2 does not beat E1 by ≥ 0.3 ms, the GEMV kernel is wrong about where the time goes.

## Known limits, stated up front
Batch-1/small-batch decode only: no scheduler, no continuous batching, no paged KV cache, greedy sampling, one
GPU. The comparison with vLLM is about the decode step, not about serving. At batch ≥ 16 the GEMV kernels are not
the right tool (tensor cores win) and are not used there.

## Change log

- 2026-11-14, after the first E1–E3 run and an E3 profile: added **E4** (fused RMSNorm prologue and residual epilogue
  in the GEMV kernels, RoPE + KV-cache write in one kernel), because the profile put 1.39 ms per step (29%) in ~950
  small torch kernel launches, not in the GEMVs (289–301 GB/s) or attention. Torch attention for E0–E2 changed from SDPA
  with a mask (7.44 ms per step for E1) to an explicit grouped matmul + masked softmax. GEMV for M > 1 switched to
  tensor cores after it lost to cuBLAS at batch 8. H1 uses `torch.compile` default mode: `reduce-overhead`
  segfaults in this image. vLLM reference measured offline (`vllm_ref.py`) under the same conditions.
