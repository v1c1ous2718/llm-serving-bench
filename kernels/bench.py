"""Decode speed (DESIGN.md "Metric"). Run inside the pinned image: kernels/run.sh bench.py [variants...]

Per variant, batch 1, prompt = 512 tokens of the model card, greedy:
  step_ms    median time of one decode step at position 512 (E*: 200 CUDA-graph replays; H*/E0: 64 synced steps)
  e2e_ms     generating 256 tokens after the prompt, divided by 256 (prefill excluded, argmax included)
For E1-E3 also batch 4 and 8 (step_ms only). Results append to results/bench.jsonl.
"""
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import Engine, find_model

dev, dt = "cuda", torch.bfloat16
path = find_model()
tok = AutoTokenizer.from_pretrained(path)
ids = tok(open(f"{path}/README.md").read())["input_ids"]
P, NEW = 512, 256
prompt = torch.tensor([ids[:P]], device=dev)
WEIGHT_GB = 0.988
out_rows = []


def record(row):
    row["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["gpu"] = torch.cuda.get_device_name()
    if "step_ms" in row:
        row["eff_GBps"] = round(WEIGHT_GB / row["step_ms"] * 1e3, 1)
    print(json.dumps(row), flush=True)
    out_rows.append(row)


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t, r


# ---------------------------------------------------------------- HF baselines
def hf_variant(name, compiled):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(dev).eval()
    kw = dict(attention_mask=torch.ones_like(prompt), do_sample=False, repetition_penalty=1.0, top_k=None,
              top_p=None, temperature=None)
    if compiled:
        kw["cache_implementation"] = "static"
        # mode="reduce-overhead" (CUDA graphs) segfaults in this image (exit 139, 2026-11-14) and inductor already
        # skips its graphs for the static cache ("mutated inputs"); the default mode is the working compiled path.
        model.forward = torch.compile(model.forward, mode=os.environ.get("H1_MODE", "default"))
    with torch.no_grad():
        for _ in range(2 if compiled else 1):                        # compile / warm-up
            model.generate(prompt, max_new_tokens=NEW + 1, min_new_tokens=NEW + 1, **kw)
        t1 = [sync_time(lambda: model.generate(prompt, max_new_tokens=1, **kw))[0] for _ in range(3)]
        tn = [sync_time(lambda: model.generate(prompt, max_new_tokens=NEW + 1, min_new_tokens=NEW + 1, **kw))[0]
              for _ in range(3)]
        e2e = (statistics.median(tn) - statistics.median(t1)) / NEW * 1e3
        row = {"variant": name, "batch": 1, "e2e_ms": round(e2e, 3)}
        if not compiled:                                             # per-step at position 512, eager
            from transformers import DynamicCache
            cache = DynamicCache()
            out = model(prompt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            ts = []
            for _ in range(64):
                dtm, out = sync_time(lambda: model(nxt, past_key_values=cache, use_cache=True))
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                ts.append(dtm * 1e3)
            row["step_ms"] = round(statistics.median(ts), 3)
        else:
            row["step_ms"] = row["e2e_ms"]
            row["note"] = "step_ms = e2e_ms (compiled static-cache path, average over positions 512-768)"
    record(row)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- my engine
def engine_variant(name, gemv, attn, graph, fused=False):
    eng = Engine(path, max_batch=8, max_ctx=1024, gemv=gemv, attn=attn, fused=fused)
    with torch.no_grad():
        logits = eng.prefill(prompt)
        first = logits.argmax(-1)
        if not graph:                                                # E0: eager steps
            pos = torch.tensor([P], device=dev)
            t = first
            ts = []
            for i in range(64):
                dtm, lg = sync_time(lambda: eng.step(t, pos, ctx=eng.max_ctx))
                t = lg.argmax(-1)
                pos = pos + 1
                ts.append(dtm * 1e3)
            step = statistics.median(ts)
            eng.prefill(prompt)
            pos = torch.tensor([P], device=dev)
            t = first
            state = {"t": first}

            def gen_eager():
                for _ in range(NEW):
                    state["t"] = eng.step(state["t"], pos, ctx=eng.max_ctx).argmax(-1)
                    pos.add_(1)
            dtm, _ = sync_time(gen_eager)
            record({"variant": name, "batch": 1, "step_ms": round(step, 3), "e2e_ms": round(dtm / NEW * 1e3, 3)})
            return
        rows, e2es = {}, {}
        for B in (1, 4, 8):
            eng.prefill(prompt.expand(B, P).contiguous())
            eng.capture(B, pos0=P)
            tokb = first.expand(B).contiguous()
            posb = torch.full((B,), P, device=dev)
            for _ in range(20):
                eng.replay(tokb, posb)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(200):
                s.record()
                eng.graph.replay()
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e))
            rows[B] = statistics.median(ts)
            if True:                                                 # end-to-end 256 tokens, position advancing
                eng.prefill(prompt.expand(B, P).contiguous())
                eng.g_tok.copy_(tokb)
                eng.g_pos.fill_(P)

                def gen():
                    for _ in range(NEW):
                        eng.graph.replay()
                        eng.g_tok.copy_(eng.g_out.argmax(-1))
                        eng.g_pos.add_(1)
                dtm, _ = sync_time(gen)
                e2es[B] = dtm / NEW * 1e3
        for B in (1, 4, 8):
            record({"variant": name, "batch": B, "step_ms": round(rows[B], 3), "e2e_ms": round(e2es[B], 3)})
    del eng
    torch.cuda.empty_cache()


VARIANTS = {
    "H0": lambda: hf_variant("H0", compiled=False),
    "H1": lambda: hf_variant("H1", compiled=True),
    "E0": lambda: engine_variant("E0", gemv=False, attn=False, graph=False),
    "E1": lambda: engine_variant("E1", gemv=False, attn=False, graph=True),
    "E2": lambda: engine_variant("E2", gemv=True, attn=False, graph=True),
    "E3": lambda: engine_variant("E3", gemv=True, attn=True, graph=True),
    "E4": lambda: engine_variant("E4", gemv=True, attn=True, graph=True, fused=True),
}
for v in (sys.argv[1:] or list(VARIANTS)):
    try:
        VARIANTS[v]()
    except Exception as ex:  # one failing variant must not hide the others
        record({"variant": v, "error": f"{type(ex).__name__}: {str(ex)[:300]}"})
import os
os.makedirs("results", exist_ok=True)
with open("results/bench.jsonl", "a") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")
