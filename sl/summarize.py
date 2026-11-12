"""Per-run summary of a client JSONL file: TTFT / TPOT / E2E percentiles and send lateness.

uv run python -m sl.summarize runs/x.jsonl [--warmup 50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, warmup: int) -> list[dict]:
    recs = [json.loads(l) for l in path.open()]
    recs.sort(key=lambda r: r["idx"])
    return recs[warmup:]


def summarize(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["error"] is None and r["chunks"]]
    ttft = np.array([r["chunks"][0] - r["send"] for r in ok])
    tpot = np.array([(r["end"] - r["chunks"][0]) / (r["output_tokens"] - 1)
                     for r in ok if r.get("output_tokens", 0) > 1])
    e2e = np.array([r["end"] - r["send"] for r in ok])
    late = np.array([r["send"] - r["scheduled"] for r in recs if r["scheduled"] is not None])
    span = max(r["end"] for r in ok) - min(r["send"] for r in ok)
    out_tok = sum(r.get("output_tokens", 0) for r in ok)
    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (50, 95, 99)} if len(a) else {}
    return {
        "n": len(recs), "errors": len(recs) - len(ok),
        "ttft_s": pct(ttft), "tpot_ms": {k: v * 1e3 for k, v in pct(tpot).items()}, "e2e_s": pct(e2e),
        "send_lateness_ms": {k: v * 1e3 for k, v in pct(late).items()},
        "req_per_s": len(ok) / span, "out_tok_per_s": out_tok / span,
        "short_tokens": sum(r.get("output_tokens", 0) != r["max_tokens"] for r in ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--warmup", type=int, default=50)
    a = ap.parse_args()
    print(json.dumps(summarize(load(a.path, a.warmup)), indent=1))


if __name__ == "__main__":
    main()
