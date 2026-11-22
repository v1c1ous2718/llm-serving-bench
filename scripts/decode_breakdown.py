"""Numbers behind RESULTS.md "Why batch-1 decode reaches 60% of the bound" and "Why throughput falls past 64".

  uv run python scripts/decode_breakdown.py runs/_profile_r1/traces runs/F0-r2

1. Batch-1 trace (c1_*.json): per engine step, wall period, GPU busy fraction, and time per kernel group.
   Weight GEMMs are matched to projections by launch order inside a layer (qkv, o, gate_up, down) and
   lm_head is the single large GEMV between steps.
2. Batch-256 trace (c256_*.json): steps with and without a prefill in the batch.
3. Without the profiler: inter-token gaps of every request in the closed-loop F0 runs (c = 64, 128, 256).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

H, I, V = 896, 4864, 151936
QKV = H + 256
MB = {"qkv": QKV * H * 2e-6, "o_proj": H * H * 2e-6, "gate_up": 2 * I * H * 2e-6, "down": H * I * 2e-6,
      "lm_head": V * H * 2e-6}


def events(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["traceEvents"]


def union(iv):
    tot, cs, ce = 0.0, None, None
    for a, b in sorted(iv):
        if ce is None or a > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + (ce - cs if ce is not None else 0.0)


def group(name: str) -> str:
    n = name.lower()
    if "flash" in n:
        return "attention"
    if "gemv" in n or "gemm" in n or "cutlass" in n or "splitk" in n:
        return "gemm"
    return "other"


def batch1(path: str) -> dict:
    ev = events(path)
    steps = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"])
    k = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy")), key=lambda e: e["ts"])
    steps = steps[10:-10]
    period = np.diff([s["ts"] for s in steps])
    t0, t1 = steps[0]["ts"], steps[-1]["ts"]
    inwin = [x for x in k if t0 <= x["ts"] < t1]
    busy = union([(x["ts"], x["ts"] + x["dur"]) for x in inwin])
    n = len(steps) - 1
    proj = Counter()
    groups = Counter()
    for x in inwin:
        g = group(x["name"])
        groups[g] += x["dur"] / n
    # Per-layer GEMM order inside one step: qkv, o_proj, gate_up, down (+ its split-K reduce).
    order = ["qkv", "o_proj", "gate_up", "down", "down"]
    for s in steps[:-1]:
        gem = [x for x in k if s["ts"] <= x["ts"] < s["ts"] + s["dur"] and group(x["name"]) == "gemm"]
        for i, x in enumerate(gem):
            proj[order[i % 5]] += x["dur"] / n
    lm = [x["dur"] for x in inwin if x["dur"] > 500]
    proj["lm_head"] = float(np.mean(lm)) if lm else float("nan")
    for p in ("qkv", "o_proj", "gate_up", "down"):
        proj[p] /= 1  # already per step (24 layers summed)
    gemm_layers = sum(proj[p] for p in ("qkv", "o_proj", "gate_up", "down"))
    return {
        "period_us": float(np.median(period)), "busy_frac": busy / (t1 - t0),
        "per_step_us": {g: round(v) for g, v in groups.items()},
        "proj_us_per_step": {p: round(v, 1) for p, v in proj.items()},
        "proj_GBps": {p: round(MB[p] * (24 if p != "lm_head" else 1) / proj[p] * 1e3)
                      for p in MB if proj[p] > 0},
        "layer_gemm_GBps": round(sum(MB[p] for p in ("qkv", "o_proj", "gate_up", "down")) * 24 / gemm_layers * 1e3),
    }


def batch256(path: str) -> dict:
    ev = events(path)
    steps = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"])
    k = [e for e in ev if e.get("cat") == "kernel"]
    out = {"decode_only": [], "with_prefill": []}
    for s in steps:
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)", s["name"])
        g = Counter()
        for x in k:
            if s["ts"] <= x["ts"] < s["ts"] + s["dur"]:
                g[group(x["name"])] += x["dur"]
        out["with_prefill" if int(m[1]) else "decode_only"].append(
            {"prefill_tokens": int(m[2]), "decodes": int(m[3]), "span_us": s["dur"], **g})
    return {key: {"n": len(v), "median_span_ms": float(np.median([r["span_us"] for r in v])) / 1e3,
                  "median_attention_ms": float(np.median([r.get("attention", 0) for r in v])) / 1e3,
                  "median_gemm_ms": float(np.median([r.get("gemm", 0) for r in v])) / 1e3}
            for key, v in out.items() if v}


def gaps(f0: Path) -> dict:
    res = {}
    for c in (64, 128, 256):
        p = f0 / f"c{c}" / "W1" / "0.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(line) for line in p.open()]
        g = np.concatenate([np.diff(r["chunks"]) for r in recs if len(r["chunks"]) > 2]) * 1e3
        fast = g[g < 2 * np.percentile(g, 30)]
        res[c] = {"p50_ms": float(np.median(g)), "mean_ms": float(g.mean()),
                  "decode_only_gap_ms": float(np.median(fast)),
                  "slow_frac": float((g >= 2 * np.percentile(g, 30)).mean()),
                  "share_of_time_in_slow_gaps": float(g[g >= 2 * np.percentile(g, 30)].sum() / g.sum())}
    return res


def main() -> None:
    traces, f0 = sys.argv[1], Path(sys.argv[2])
    res = {"batch1": batch1(glob.glob(f"{traces}/c1_*.json")[0])}
    b256 = glob.glob(f"{traces}/c256_*.json")
    if b256:
        res["batch256_profiled"] = batch256(b256[0])
    res["inter_token_gaps_no_profiler"] = gaps(f0)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
