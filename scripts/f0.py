"""F0 roofline check (DESIGN.md), rerun in one session on a quiet GPU.

  sg docker -c "uv run python scripts/f0.py F0-r2"        # run, then print the table
  uv run python scripts/f0.py --table runs/F0              # table only, for any F0 directory
  F0_CONC=64,128,256 uv run python scripts/f0.py F0-triton --attention-backend TRITON_ATTN   # extra server args

Same points as the first F0 run: W1, closed loop, seed 0, N = max(120, 4c) (batch 1: 60), server at
--gpu-memory-utilization 0.85. Every point waits for a quiet GPU and samples /metrics + nvidia-smi at 1 Hz
(scripts/sweep.py point()).

Steady-state throughput: output chunks whose timestamp falls inside the window in which all c slots are busy,
i.e. from the last first-token among the first c requests to the send of the last request, divided by its
length. This excludes ramp-up and drain, which the first F0 table did not do consistently.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sl.summarize import load, summarize  # noqa: E402
import sweep  # noqa: E402

import os
CONC = [int(c) for c in os.environ.get("F0_CONC", "1,2,4,8,16,32,64,128,256").split(",")]


def steady_tok_s(recs: list[dict], c: int) -> float:
    recs = sorted(recs, key=lambda r: r["idx"])
    ok = [r for r in recs if r["error"] is None and r["chunks"]]
    t0 = max(r["chunks"][0] for r in ok[:c])
    t1 = max(r["send"] for r in ok)
    if t1 - t0 < 1.0:
        return float("nan")
    n = sum(t0 <= t < t1 for r in ok for t in r["chunks"])
    return n / (t1 - t0)


def table(d: Path) -> list[dict]:
    rows = []
    for c in CONC:
        p = d / ("bs1" if c == 1 else f"c{c}") / "W1" / "0.jsonl"
        if not p.exists():
            continue
        all_recs = load(p, 0)
        s = summarize(load(p, 10 if c == 1 else c))
        rows.append({"c": c, "tok_s": steady_tok_s(all_recs, c), "tpot_p50": s["tpot_ms"]["p50"],
                     "tpot_p95": s["tpot_ms"]["p95"], "ttft_p50_ms": s["ttft_s"]["p50"] * 1e3})
    end = d / "bs1_end" / "W1" / "0.jsonl"
    if end.exists():
        rows.append({"c": "1 (end)", "tpot_p50": summarize(load(end, 10))["tpot_ms"]["p50"]})
    return rows


def run(name: str, extra: list[str]) -> None:
    d = sweep.RUNS / name
    sweep.log(f"{name}: starting server (F0, 0.85)")
    subprocess.run([str(ROOT / "scripts/serve.sh"), name, "--gpu-memory-utilization", "0.85", *extra], cwd=ROOT,
                   check=True)
    for c in [1] + [c for c in CONC if c != 1]:
        n = 60 if c == 1 else max(120, 4 * c)
        out = sweep.point(d / ("bs1" if c == 1 else f"c{c}") / "W1" / "0.jsonl", "W1", 0, n, "concurrency", c)
        s = summarize(load(out, 10 if c == 1 else c))
        sweep.log(f"{name}: c={c} TPOT p50 {s['tpot_ms']['p50']:.2f} ms, "
                  f"steady {steady_tok_s(load(out, 0), c):.0f} tok/s")
    sweep.point(d / "bs1_end" / "W1" / "0.jsonl", "W1", 0, 60, "concurrency", 1)


def main() -> None:
    if sys.argv[1] == "--table":
        d = Path(sys.argv[2])
    else:
        run(sys.argv[1], sys.argv[2:])
        d = sweep.RUNS / sys.argv[1]
    rows = table(d)
    (d / "table.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
