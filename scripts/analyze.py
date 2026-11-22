"""Recompute the sweep tables from runs/ alone (DESIGN.md "Outputs").

  uv run python scripts/analyze.py            # every session found
  uv run python scripts/analyze.py baseline   # one session

Per session, workload and rate multiple: TTFT / TPOT p50 and p95 pooled over the seeds present, the spread of the
per-seed p95 values, completed req/s, preemptions during the point, and whether both SLO terms hold.
Goodput = the highest tested rate at which both SLO terms hold (DESIGN.md "Metrics").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sl.summarize import load  # noqa: E402

RUNS = ROOT / "runs"
MULTIPLES = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
TTFT_SLO = {"W1": 1.0, "W3": 1.0, "W2": 3.0, "Wmix": 3.0}
TPOT_SLO_MS = 50.0
WARMUP = 50


def per_request(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ok = [r for r in recs if r["error"] is None and r["chunks"]]
    ttft = np.array([r["chunks"][0] - r["send"] for r in ok])
    tpot = np.array([(r["end"] - r["chunks"][0]) / (r["output_tokens"] - 1) * 1e3
                     for r in ok if r.get("output_tokens", 0) > 1])
    return ttft, tpot


def preemptions(metrics: Path) -> float:
    if not metrics.exists():
        return float("nan")
    rows = [json.loads(line) for line in metrics.open()]
    vals = [r["num_preemptions_total"] for r in rows if "num_preemptions_total" in r]
    return vals[-1] - vals[0] if len(vals) > 1 else 0.0


def video_seconds(metrics: Path) -> int:
    """Seconds during the point in which the video decoder/encoder was active (another application on the GPU;
    the runner checks only before each point)."""
    if not metrics.exists():
        return 0
    rows = [json.loads(line) for line in metrics.open()]
    return sum(r.get("dec_pct", 0) > 0 or r.get("enc_pct", 0) > 0 for r in rows)


def point(session: str, w: str, m: float, cap: float) -> dict | None:
    files = sorted((RUNS / session / f"x{m}" / w).glob("[0-9].jsonl"))
    if not files and session == "baseline":  # the high multiples ran in their own session
        files = sorted((RUNS / "baseline-hi" / f"x{m}" / w).glob("[0-9].jsonl"))
    if not files:
        return None
    ttft_all, tpot_all, p95s, errors, pre = [], [], [], 0, 0.0
    for f in files:
        recs = load(f, WARMUP)
        errors += sum(r["error"] is not None for r in recs)
        t, p = per_request(recs)
        ttft_all.append(t)
        tpot_all.append(p)
        p95s.append((np.percentile(t, 95), np.percentile(p, 95)))
        pre += preemptions(f.with_suffix(".metrics.jsonl"))
        if (v := video_seconds(f.with_suffix(".metrics.jsonl"))):
            print(f"WARNING {f.relative_to(RUNS)}: video decoder active {v} s during the point")
    t, p = np.concatenate(ttft_all), np.concatenate(tpot_all)
    row = {"session": session, "workload": w, "x": m, "rate": round(m * cap, 3), "seeds": len(files),
           "ttft_p50": float(np.median(t)), "ttft_p95": float(np.percentile(t, 95)),
           "tpot_p50": float(np.median(p)), "tpot_p95": float(np.percentile(p, 95)),
           "ttft_p95_seed_range": [float(min(a for a, _ in p95s)), float(max(a for a, _ in p95s))],
           "tpot_p95_seed_range": [float(min(b for _, b in p95s)), float(max(b for _, b in p95s))],
           "errors": errors, "preemptions": pre}
    # DESIGN.md: goodput uses the two latency terms only; errors are reported in their own column.
    row["slo_ok"] = row["ttft_p95"] <= TTFT_SLO[w] and row["tpot_p95"] <= TPOT_SLO_MS
    return row


def main() -> None:
    cap = json.loads((RUNS / "capacity.json").read_text())
    sessions = sys.argv[1:] or sorted(d.name for d in RUNS.iterdir()
                                      if d.is_dir() and not d.name.startswith(("_", "F0")) and d.name != "baseline-hi")
    out = []
    for s in sessions:
        for w in ("W1", "W2", "Wmix", "W3"):
            rows = [r for m in MULTIPLES if (r := point(s, w, m, cap.get(w, float("nan"))))]
            if not rows:
                continue
            ok = [r["rate"] for r in rows if r["slo_ok"]]
            # Goodput: highest rate below which every tested rate also meets the SLO.
            good = None
            for r in rows:
                if not r["slo_ok"]:
                    break
                good = r["rate"]
            print(f"\n{s} {w}: goodput {good} req/s (seeds per point: {sorted({r['seeds'] for r in rows})})")
            print(f"{'x':>5} {'req/s':>7} {'TTFT p50':>9} {'TTFT p95':>9} {'TPOT p50':>9} {'TPOT p95':>9} "
                  f"{'preempt':>7} {'err':>3}  SLO")
            for r in rows:
                print(f"{r['x']:>5} {r['rate']:>7.2f} {r['ttft_p50']:>8.3f}s {r['ttft_p95']:>8.3f}s "
                      f"{r['tpot_p50']:>7.1f}ms {r['tpot_p95']:>7.1f}ms {r['preemptions']:>7.0f} {r['errors']:>3}  "
                      f"{'ok' if r['slo_ok'] else 'FAIL'}")
            out.append({"session": s, "workload": w, "goodput": good, "any_ok": ok, "points": rows})
    (RUNS / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
