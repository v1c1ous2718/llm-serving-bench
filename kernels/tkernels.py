"""My Triton kernels for small-batch decode (DESIGN.md E2, E3).

gemv(x, w, bias)            y[M, N] = x[M, K] @ w[N, K]^T (+ bias), M <= 8, BF16 in / BF16 out, FP32 accumulate.
gemv_silu_mul(x, w_gu)      y[M, I] = silu(x @ w_gate^T) * (x @ w_up^T), w_gu = cat(w_gate, w_up) along N.
decode_attention(q, k_cache, v_cache, seq_len, out)
                            GQA flash-decoding: q[B, H, D], caches [B, KVH, T, D], seq_len[B] on device.

The weight matrix is streamed exactly once per call: each program owns BLOCK_N output rows and walks K in BLOCK_K
steps, so loads are contiguous 2*BLOCK_K-byte rows. M <= 8 rows of x are kept in registers and broadcast.
Configs are autotuned per (M, N, K); the best one is cached by Triton.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _gemv_configs():
    cfgs = []
    for bn in (4, 8, 16, 32):
        for bk in (128, 256, 512):
            for nw in (2, 4, 8):
                cfgs.append(triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw))
    return cfgs


def _prune(configs, named_args, **kw):
    if named_args.get("M", 1) > 1 or kw.get("M", 1) > 1:
        return [c for c in configs if c.kwargs["BLOCK_N"] >= 16 and c.kwargs["BLOCK_K"] <= 256]
    return configs


@triton.autotune(configs=_gemv_configs(), key=["M", "N", "K"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemv_kernel(x_ptr, w_ptr, b_ptr, y_ptr, ln_ptr, r_ptr, eps, M, N, K, stride_xm, stride_wn, stride_ym,
                 HAS_BIAS: tl.constexpr, NORM: tl.constexpr, RESID: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M
    if NORM:  # RMSNorm prologue: every program recomputes rstd of its M rows (K <= 4864, from L2)
        ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                         mask=mask_m[:, None] & (offs_k < K)[None, :], other=0.0).to(tl.float32)
            ss += tl.sum(xv * xv, axis=1)
        rstd = 1.0 / tl.sqrt(ss / K + eps)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)        # [BN, BK]
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)        # [BM, BK]
        if NORM:  # match the torch reference: normalize, round to bf16, scale by the weight, round again
            g = tl.load(ln_ptr + offs_k, mask=mask_k, other=0.0)
            x = ((x * rstd[:, None]).to(tl.bfloat16) * g[None, :]).to(tl.float32)
        if BLOCK_M == 1:
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)                             # [1, BN], CUDA cores
        else:                                                                                # M > 1: tensor cores
            acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    ymask = mask_m[:, None] & mask_n[None, :]
    if RESID:  # residual epilogue (r may alias y: each program reads and writes only its own columns)
        acc += tl.load(r_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], mask=ymask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=ymask)


@triton.autotune(configs=_gemv_configs(), key=["M", "I", "K"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemv_silu_mul_kernel(x_ptr, w_ptr, y_ptr, ln_ptr, eps, M, I, K, stride_xm, stride_wn, stride_ym,
                          NORM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_n = offs_n < I
    mask_m = offs_m < M
    if NORM:
        ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                         mask=mask_m[:, None] & (offs_k < K)[None, :], other=0.0).to(tl.float32)
            ss += tl.sum(xv * xv, axis=1)
        rstd = 1.0 / tl.sqrt(ss / K + eps)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        wmask = mask_n[:, None] & mask_k[None, :]
        wg = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=wmask, other=0.0).to(tl.float32)
        wu = tl.load(w_ptr + (offs_n[:, None] + I) * stride_wn + offs_k[None, :], mask=wmask,
                     other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        if NORM:
            g = tl.load(ln_ptr + offs_k, mask=mask_k, other=0.0)
            x = ((x * rstd[:, None]).to(tl.bfloat16) * g[None, :]).to(tl.float32)
        if BLOCK_M == 1:
            acc_g += tl.sum(x[:, None, :] * wg[None, :, :], axis=2)
            acc_u += tl.sum(x[:, None, :] * wu[None, :, :], axis=2)
        else:
            xb = x.to(tl.bfloat16)
            acc_g += tl.dot(xb, tl.trans(wg.to(tl.bfloat16)))
            acc_u += tl.dot(xb, tl.trans(wu.to(tl.bfloat16)))
    # match torch: gate and up are rounded to bf16 before SiLU and the product
    g_b = acc_g.to(tl.bfloat16).to(tl.float32)
    u_b = acc_u.to(tl.bfloat16).to(tl.float32)
    y = (g_b / (1.0 + tl.exp(-g_b))).to(tl.bfloat16).to(tl.float32) * u_b
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], y.to(y_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def _block_m(m: int) -> int:
    """M = 1: broadcast-multiply-reduce on CUDA cores. 2 <= M <= 8: rows padded to 16 for tl.dot (tensor cores);
    the padding costs compute, not memory traffic, which is what bounds these shapes."""
    assert 1 <= m <= 8, "GEMV kernels are for M <= 8"
    return 1 if m == 1 else 16


def gemv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None, out: torch.Tensor | None = None,
         norm: torch.Tensor | None = None, eps: float = 1e-6, residual: torch.Tensor | None = None) -> torch.Tensor:
    """y = (rmsnorm(x) * norm if norm is given else x) @ w^T + bias + residual. out may be residual (in place)."""
    M, K = x.shape
    N = w.shape[0]
    assert w.stride(1) == 1 and x.stride(1) == 1
    y = out if out is not None else torch.empty((M, N), device=x.device, dtype=x.dtype)
    if residual is not None:
        assert residual.stride() == y.stride()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_kernel[grid](x, w, bias if bias is not None else w, y, norm if norm is not None else w,
                       residual if residual is not None else y, eps, M, N, K, x.stride(0), w.stride(0), y.stride(0),
                       HAS_BIAS=bias is not None, NORM=norm is not None, RESID=residual is not None,
                       BLOCK_M=_block_m(M))
    return y


def gemv_silu_mul(x: torch.Tensor, w_gu: torch.Tensor, out: torch.Tensor | None = None,
                  norm: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    M, K = x.shape
    I = w_gu.shape[0] // 2
    y = out if out is not None else torch.empty((M, I), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(I, meta["BLOCK_N"]),)
    _gemv_silu_mul_kernel[grid](x, w_gu, y, norm if norm is not None else w_gu, eps, M, I, K, x.stride(0),
                                w_gu.stride(0), y.stride(0), NORM=norm is not None, BLOCK_M=_block_m(M))
    return y


# ---------------------------------------------------------------- RoPE + KV-cache write, one kernel per layer

@triton.jit
def _rope_cache_kernel(qkv_ptr, pos_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr,
                       stride_qkv, stride_qb, stride_qh, stride_kb, stride_kh, stride_kt,
                       H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)                     # 0..H-1: query heads, H..H+KV-1: key heads, H+KV..: value heads
    half = tl.arange(0, D // 2)
    pos = tl.load(pos_ptr + b)
    base = qkv_ptr + b * stride_qkv + h * D
    if h < H + KV:
        x1 = tl.load(base + half).to(tl.float32)
        x2 = tl.load(base + D // 2 + half).to(tl.float32)
        c = tl.load(cos_ptr + pos * (D // 2) + half)
        s = tl.load(sin_ptr + pos * (D // 2) + half)
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        if h < H:
            dst = q_ptr + b * stride_qb + h * stride_qh
        else:
            dst = k_ptr + b * stride_kb + (h - H) * stride_kh + pos * stride_kt
        tl.store(dst + half, o1.to(tl.bfloat16))
        tl.store(dst + D // 2 + half, o2.to(tl.bfloat16))
    else:
        dst = v_ptr + b * stride_kb + (h - H - KV) * stride_kh + pos * stride_kt
        offs = tl.arange(0, D)
        tl.store(dst + offs, tl.load(base + offs))


def rope_cache(qkv, pos, cos, sin, q_out, k_cache, v_cache, H, KV, D):
    """qkv [B, (H+2KV)*D] -> q_out [B, H, D] rotated; rotated k and raw v written at pos into the caches."""
    B = qkv.shape[0]
    _rope_cache_kernel[(B, H + 2 * KV)](qkv, pos, cos, sin, q_out, k_cache, v_cache,
                                        qkv.stride(0), q_out.stride(0), q_out.stride(1),
                                        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                                        H=H, KV=KV, D=D, num_warps=1)


# ---------------------------------------------------------------- decode attention (flash-decoding, GQA-aware)

@triton.jit
def _attn_split_kernel(q_ptr, k_ptr, v_ptr, len_ptr, part_o_ptr, part_ml_ptr, sm_scale,
                       stride_qb, stride_qh, stride_kb, stride_kh, stride_kt,
                       stride_pb, stride_ph, stride_ps,
                       GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, D: tl.constexpr,
                       SPLITS: tl.constexpr, BLOCK_T: tl.constexpr):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)
    seq_len = tl.load(len_ptr + b)
    chunk = tl.cdiv(tl.cdiv(seq_len, SPLITS), BLOCK_T) * BLOCK_T
    t_start = s * chunk
    t_end = tl.minimum(t_start + chunk, seq_len)

    offs_g = tl.arange(0, GROUP_PAD)
    offs_d = tl.arange(0, D)
    heads = kvh * GROUP + offs_g
    mask_g = offs_g < GROUP
    q = tl.load(q_ptr + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
                mask=mask_g[:, None], other=0.0)                                               # [G, D] bf16

    m_i = tl.full((GROUP_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GROUP_PAD,), dtype=tl.float32)
    acc = tl.zeros((GROUP_PAD, D), dtype=tl.float32)
    kbase = k_ptr + b * stride_kb + kvh * stride_kh
    vbase = v_ptr + b * stride_kb + kvh * stride_kh
    for t0 in range(t_start, t_end, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < t_end
        k = tl.load(kbase + offs_t[:, None] * stride_kt + offs_d[None, :], mask=mask_t[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * sm_scale                                             # [G, BT] fp32
        scores = tl.where(mask_t[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vbase + offs_t[:, None] * stride_kt + offs_d[None, :], mask=mask_t[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    # Empty split (t_start >= seq_len): m = -inf, l = 0; the combine kernel gives it zero weight.
    po = part_o_ptr + b * stride_pb + heads[:, None] * stride_ph + s * stride_ps + offs_d[None, :]
    tl.store(po, acc, mask=mask_g[:, None])
    pml = part_ml_ptr + (b * (tl.num_programs(1) * GROUP) + heads) * SPLITS * 2 + s * 2
    tl.store(pml, m_i, mask=mask_g)
    tl.store(pml + 1, l_i, mask=mask_g)


@triton.jit
def _attn_combine_kernel(part_o_ptr, part_ml_ptr, out_ptr, H,
                         stride_pb, stride_ph, stride_ps, stride_ob, stride_oh,
                         D: tl.constexpr, SPLITS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, D)
    ml = part_ml_ptr + (b * H + h) * SPLITS * 2 + offs_s * 2
    m = tl.load(ml)
    l = tl.load(ml + 1)
    m_max = tl.max(m, axis=0)
    w = tl.where(m > float("-inf"), tl.exp(m - m_max), 0.0)
    o = tl.load(part_o_ptr + b * stride_pb + h * stride_ph + offs_s[:, None] * stride_ps + offs_d[None, :])
    num = tl.sum(o * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, (num / den).to(out_ptr.dtype.element_ty))


class DecodeAttention:
    """Holds the split buffers so the call is CUDA-graph safe (no allocation, lengths read on device)."""

    def __init__(self, max_batch: int, n_heads: int, n_kv: int, head_dim: int, splits: int = 16,
                 block_t: int = 32, device: str = "cuda"):
        self.H, self.KV, self.D, self.S, self.BT = n_heads, n_kv, head_dim, splits, block_t
        self.group = n_heads // n_kv
        self.part_o = torch.empty((max_batch, n_heads, splits, head_dim), device=device, dtype=torch.float32)
        self.part_ml = torch.empty((max_batch, n_heads, splits, 2), device=device, dtype=torch.float32)
        self.scale = head_dim ** -0.5

    def __call__(self, q, k_cache, v_cache, seq_len, out):
        B = q.shape[0]
        _attn_split_kernel[(B, self.KV, self.S)](
            q, k_cache, v_cache, seq_len, self.part_o, self.part_ml, self.scale,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.part_o.stride(0), self.part_o.stride(1), self.part_o.stride(2),
            GROUP=self.group, GROUP_PAD=max(16, triton.next_power_of_2(self.group)), D=self.D,
            SPLITS=self.S, BLOCK_T=self.BT, num_warps=4)
        _attn_combine_kernel[(B, self.H)](
            self.part_o, self.part_ml, out, self.H,
            self.part_o.stride(0), self.part_o.stride(1), self.part_o.stride(2), out.stride(0), out.stride(1),
            D=self.D, SPLITS=self.S, num_warps=2)
        return out
