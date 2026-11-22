"""Correctness (DESIGN.md "Correctness"). Run inside the pinned image:
  kernels/run.sh check.py
1. Each Triton kernel vs torch on the model's real shapes, M = 1, 4, 8.
2. Each engine variant vs HF eager: greedy tokens for 8 prompts × 64 tokens, first-step max |logit diff|.
"""
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import tkernels as tk
from engine import Engine, find_model

torch.manual_seed(0)
res = {"kernels": [], "engine": {}}
dev, dt = "cuda", torch.bfloat16

for (N, K) in [(1152, 896), (896, 896), (896, 4864), (151936, 896)]:
    w = torch.randn(N, K, device=dev, dtype=dt) * 0.02
    b = torch.randn(N, device=dev, dtype=dt) * 0.02
    for M in (1, 4, 8):
        x = torch.randn(M, K, device=dev, dtype=dt)
        ref = (x.float() @ w.float().t() + b.float())
        got = tk.gemv(x, w, b).float()
        res["kernels"].append({"kernel": "gemv", "M": M, "N": N, "K": K,
                               "max_abs_err": float((got - ref).abs().max()), "ref_max": float(ref.abs().max())})
wgu = torch.randn(2 * 4864, 896, device=dev, dtype=dt) * 0.02
for M in (1, 4, 8):
    x = torch.randn(M, 896, device=dev, dtype=dt)
    g, u = (x.float() @ wgu.float().t()).chunk(2, dim=-1)
    ref = F.silu(g) * u
    got = tk.gemv_silu_mul(x, wgu).float()
    res["kernels"].append({"kernel": "gemv_silu_mul", "M": M, "max_abs_err": float((got - ref).abs().max()),
                           "ref_max": float(ref.abs().max())})
att = tk.DecodeAttention(8, 14, 2, 64)
for B, T in [(1, 1), (1, 37), (1, 513), (4, 300), (8, 1000)]:
    q = torch.randn(B, 14, 64, device=dev, dtype=dt)
    kc = torch.randn(B, 2, 1024, 64, device=dev, dtype=dt)
    vc = torch.randn(B, 2, 1024, 64, device=dev, dtype=dt)
    lens = torch.randint(1, T + 1, (B,), device=dev)
    lens[0] = T
    out = torch.empty(B, 14, 64, device=dev, dtype=dt)
    att(q, kc, vc, lens, out)
    errs = []
    for i in range(B):
        L = int(lens[i])
        r = F.scaled_dot_product_attention(q[i:i + 1, :, None].float(), kc[i:i + 1, :, :L].float(),
                                           vc[i:i + 1, :, :L].float(), enable_gqa=True)[0, :, 0]
        errs.append(float((out[i].float() - r).abs().max()))
    res["kernels"].append({"kernel": "decode_attention", "B": B, "T": T, "max_abs_err": max(errs)})

# ---------------------------------------------------------------- engine vs HF
path = find_model()
hf = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(dev).eval()
# Real text (the model card): random token IDs give flat distributions where BF16 noise flips the argmax.
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(path)(open(f"{path}/README.md").read())["input_ids"]
assert len(ids) >= 8 * 64, len(ids)
prompts = torch.tensor([ids[i * 64:(i + 1) * 64] for i in range(8)], device=dev)
NEW = 64
with torch.no_grad():
    # The model's generation_config sets repetition_penalty 1.1, which HF applies even with do_sample=False;
    # plain greedy argmax is the reference here.
    hf_out = hf.generate(prompts, attention_mask=torch.ones_like(prompts), max_new_tokens=NEW, do_sample=False,
                         min_new_tokens=NEW, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)[:, 64:]
    hf_first = hf(prompts).logits[:, -1].float()
    top2 = hf_first.topk(2, dim=-1).values
    res["hf_first_step_top1_minus_top2"] = [round(float(v), 3) for v in (top2[:, 0] - top2[:, 1])]
del hf
torch.cuda.empty_cache()

for name, kw in {"E0": {}, "E2": {"gemv": True}, "E3": {"gemv": True, "attn": True},
                 "E4": {"fused": True}}.items():
    eng = Engine(path, max_batch=8, max_ctx=256, **kw)
    logits = eng.prefill(prompts)
    first_diff = float((logits.float() - hf_first).abs().max())
    toks = [logits.argmax(-1)]
    pos = torch.full((8,), 64, device=dev)
    for t in range(NEW - 1):
        logits = eng.step(toks[-1], pos, ctx=int(pos[0]) + 1)
        toks.append(logits.argmax(-1))
        pos = pos + 1
    mine = torch.stack(toks, 1)
    same = (mine == hf_out)
    first_div = [int((~same[i]).nonzero()[0]) if (~same[i]).any() else NEW for i in range(8)]
    res["engine"][name] = {"first_step_max_logit_diff": first_diff, "tokens_matching_before_divergence": first_div,
                           "all_match": int(sum(d == NEW for d in first_div))}
    del eng
    torch.cuda.empty_cache()

print(json.dumps(res, indent=1))
