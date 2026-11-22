"""My Qwen2 decode engine (DESIGN.md E0–E3).

Engine(path, max_batch, max_ctx, gemv=False, attn=False)
  .prefill(tokens[B, T])          fills the static KV cache (torch ops, causal SDPA); returns last logits [B, V]
  .step(tokens[B], pos[B])        one decode step, eager; returns logits [B, V]
  .capture(B)                     captures step() for batch B in a CUDA graph (inputs/outputs are static buffers)
  .replay(tokens, pos)            copies inputs, replays the graph, returns the static logits buffer

gemv=True  -> every weight matrix goes through tkernels.gemv / gemv_silu_mul (E2)
attn=True  -> decode attention goes through tkernels.DecodeAttention (E3)
With attn=False the decode attention is torch SDPA over the cache slice [:ctx] with a length mask; capture() uses
ctx = max_ctx so one graph serves every position (it reads up to max_ctx keys: 24 x 2 x 2 x 64 x 2 B = 12 KiB per
key position, ~0.04 ms per step at 1,024 on this card, stated in RESULTS).
"""
from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import tkernels as tk


def find_model(name="Qwen/Qwen2.5-0.5B-Instruct") -> str:
    root = os.path.expanduser("~/.cache/huggingface/hub/models--" + name.replace("/", "--"))
    return sorted(glob.glob(f"{root}/snapshots/*"))[-1]


class Engine:
    def __init__(self, path: str, max_batch: int = 8, max_ctx: int = 1024, gemv: bool = False, attn: bool = False,
                 fused: bool = False):
        cfg = json.load(open(f"{path}/config.json"))
        self.H, self.KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hid, self.inter, self.L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
        self.D = self.hid // self.H
        self.eps, self.theta = cfg["rms_norm_eps"], cfg["rope_theta"]
        self.use_gemv, self.use_attn, self.fused = gemv or fused, attn or fused, fused
        dev, dt = "cuda", torch.bfloat16
        sd = load_file(f"{path}/model.safetensors", device=dev)
        g = lambda k: sd[k].to(dt).contiguous()
        self.embed = g("model.embed_tokens.weight")
        self.lm_head = self.embed                                     # tied
        self.final_norm = g("model.norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": g(p + "input_layernorm.weight"), "ln2": g(p + "post_attention_layernorm.weight"),
                "w_qkv": torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"]).contiguous(),
                "b_qkv": torch.cat([g(p + f"self_attn.{n}_proj.bias") for n in "qkv"]).contiguous(),
                "w_o": g(p + "self_attn.o_proj.weight"),
                "w_gu": torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]).contiguous(),
                "w_down": g(p + "mlp.down_proj.weight"),
            })
        del sd
        self.weight_bytes = sum(t.numel() * 2 for l in self.layers for k, t in l.items()) + \
            self.lm_head.numel() * 2 + self.final_norm.numel() * 2
        self.max_batch, self.max_ctx = max_batch, max_ctx
        self.k_cache = torch.zeros((self.L, max_batch, self.KV, max_ctx, self.D), device=dev, dtype=dt)
        self.v_cache = torch.zeros_like(self.k_cache)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.D, 2, device=dev, dtype=torch.float32) / self.D))
        ang = torch.arange(max_ctx, device=dev, dtype=torch.float32)[:, None] * inv[None, :]
        self.cos, self.sin = ang.cos(), ang.sin()                     # [T, D/2] fp32
        self.dattn = tk.DecodeAttention(max_batch, self.H, self.KV, self.D) if self.use_attn else None
        self.graph = None

    # ------------------------------------------------------------------ pieces
    def _rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def _rope(self, x, pos):                                          # x [..., T or 1, D], pos broadcastable [.., T]
        cos, sin = self.cos[pos].unsqueeze(-3), self.sin[pos].unsqueeze(-3)   # [B, 1, T, D/2]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)

    def _lin(self, x, w, b=None):
        if self.use_gemv and x.dim() == 2 and x.shape[0] <= 8:
            return tk.gemv(x, w, b)
        return F.linear(x, w, b)

    def _mlp(self, x, l):
        if self.use_gemv and x.dim() == 2 and x.shape[0] <= 8:
            return self._lin(tk.gemv_silu_mul(x, l["w_gu"]), l["w_down"])
        gu = F.linear(x, l["w_gu"])
        g, u = gu.chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, l["w_down"])

    # ------------------------------------------------------------------ prefill (torch only, not timed)
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)[None, :].expand(B, T)
        x = self.embed[tokens]
        for i, l in enumerate(self.layers):
            h = self._rms(x, l["ln1"])
            qkv = F.linear(h, l["w_qkv"], l["b_qkv"])
            q, k, v = qkv.split([self.H * self.D, self.KV * self.D, self.KV * self.D], dim=-1)
            q = self._rope(q.view(B, T, self.H, self.D).transpose(1, 2), pos)
            k = self._rope(k.view(B, T, self.KV, self.D).transpose(1, 2), pos)
            v = v.view(B, T, self.KV, self.D).transpose(1, 2)
            self.k_cache[i, :B, :, :T] = k
            self.v_cache[i, :B, :, :T] = v
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + F.linear(o.transpose(1, 2).reshape(B, T, self.hid), l["w_o"])
            x = x + self._mlp(self._rms(x, l["ln2"]), l)
        return F.linear(self._rms(x[:, -1], self.final_norm), self.lm_head)

    # ------------------------------------------------------------------ decode step
    @torch.no_grad()
    def step(self, tokens: torch.Tensor, pos: torch.Tensor, ctx: int | None = None) -> torch.Tensor:
        """tokens [B], pos [B] (device). ctx = Python int upper bound for the torch attention slice (pos.max()+1)."""
        if self.fused:
            return self._step_fused(tokens, pos)
        B = tokens.shape[0]
        bidx = torch.arange(B, device=tokens.device)
        x = self.embed[tokens]                                        # [B, hid]
        seq_len = pos + 1
        for i, l in enumerate(self.layers):
            h = self._rms(x, l["ln1"])
            qkv = self._lin(h, l["w_qkv"], l["b_qkv"])
            q, k, v = qkv.split([self.H * self.D, self.KV * self.D, self.KV * self.D], dim=-1)
            q = self._rope(q.view(B, self.H, 1, self.D), pos[:, None]).view(B, self.H, self.D)
            k = self._rope(k.view(B, self.KV, 1, self.D), pos[:, None]).view(B, self.KV, self.D)
            self.k_cache[i, bidx, :, pos] = k
            self.v_cache[i, bidx, :, pos] = v.view(B, self.KV, self.D)
            if self.use_attn:
                o = torch.empty((B, self.H, self.D), device=x.device, dtype=x.dtype)
                self.dattn(q, self.k_cache[i, :B], self.v_cache[i, :B], seq_len, o)
            else:  # plain torch GQA attention: group the 7 query heads of each KV head, masked softmax
                T = ctx if ctx is not None else int(pos.max()) + 1
                kc, vc = self.k_cache[i, :B, :, :T], self.v_cache[i, :B, :, :T]
                mask = (torch.arange(T, device=x.device)[None, :] < seq_len[:, None])[:, None, None, :]
                sc = torch.matmul(q.view(B, self.KV, self.H // self.KV, self.D), kc.transpose(-1, -2)).float()
                sc = sc.masked_fill(~mask, float("-inf")) * self.D ** -0.5
                o = torch.matmul(torch.softmax(sc, dim=-1).to(x.dtype), vc)
            x = x + self._lin(o.reshape(B, self.hid), l["w_o"])
            x = x + self._mlp(self._rms(x, l["ln2"]), l)
        return self._lin(self._rms(x, self.final_norm), self.lm_head)

    def _step_fused(self, tokens, pos):
        """E4: 7 kernels per layer. RMSNorm in the GEMV prologue, residual add in the GEMV epilogue (in place),
        RoPE + KV-cache write in one kernel, flash-decoding attention (2 kernels)."""
        B = tokens.shape[0]
        x = self.embed[tokens]                                        # residual stream [B, hid]
        seq_len = pos + 1
        q = torch.empty((B, self.H, self.D), device=x.device, dtype=x.dtype)
        o = torch.empty_like(q)
        for i, l in enumerate(self.layers):
            qkv = tk.gemv(x, l["w_qkv"], l["b_qkv"], norm=l["ln1"], eps=self.eps)
            tk.rope_cache(qkv, pos, self.cos, self.sin, q, self.k_cache[i, :B], self.v_cache[i, :B],
                          self.H, self.KV, self.D)
            self.dattn(q, self.k_cache[i, :B], self.v_cache[i, :B], seq_len, o)
            tk.gemv(o.view(B, self.hid), l["w_o"], out=x, residual=x)
            act = tk.gemv_silu_mul(x, l["w_gu"], norm=l["ln2"], eps=self.eps)
            tk.gemv(act, l["w_down"], out=x, residual=x)
        return tk.gemv(x, self.lm_head, norm=self.final_norm, eps=self.eps)

    # ------------------------------------------------------------------ CUDA graph
    def capture(self, B: int, pos0: int = 0):
        ctx = None if self.use_attn else self.max_ctx
        self.g_tok = torch.zeros(B, dtype=torch.long, device="cuda")
        self.g_pos = torch.full((B,), pos0, dtype=torch.long, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                                        # warm-up (Triton autotune runs here)
                self.step(self.g_tok, self.g_pos, ctx)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.g_out = self.step(self.g_tok, self.g_pos, ctx)
        self.graph_ctx = ctx

    def replay(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        self.g_tok.copy_(tokens)
        self.g_pos.copy_(pos)
        self.graph.replay()
        return self.g_out
