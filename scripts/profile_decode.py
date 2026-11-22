"""Where does a decode step's time go? (RESULTS.md, F0 open questions)

  sg docker -c "uv run python scripts/profile_decode.py run"       # server + traces for c = 1, 64, 256
  uv run python scripts/profile_decode.py analyze                  # table from runs/_profile/traces

For each concurrency, a closed-loop W1 client runs until the batch is full, then vLLM's torch profiler records
PROFILE_ITERS engine iterations (CPU activities of the frontend off, stacks off, to keep overhead low).
From the trace: wall time per iteration, GPU busy time per iteration (union of kernel intervals), and the
kernel time split into GEMM / attention / other. Busy time well under wall time means the GPU idles between
kernels or between steps (CPU-side work), not that kernels are slow.
"""
from __future__ import annotations

import glob
import gzip
import json
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sweep  # noqa: E402

OUT = ROOT / "runs" / "_profile"
TRACES = OUT / "traces"
PROFILE_ITERS = 100
CONC = {1: 4, 64: 15, 256: 30}  # concurrency -> seconds to wait before profiling (batch full, decode phase)
CFG = {"profiler": "torch", "torch_profiler_dir": "/traces", "torch_profiler_with_stack": False,
       "ignore_frontend": True, "torch_profiler_use_gzip": False}


def post(path: str, body: dict | None = None) -> None:
    req = urllib.request.Request(f"http://localhost:8000{path}", method="POST",
                                 data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=300).read()
    except OSError as e:
        print(f"{path}: {e}")


def run() -> None:
    TRACES.mkdir(parents=True, exist_ok=True)
    env = {"DOCKER_ARGS": f"-v {TRACES}:/traces"}
    import os
    subprocess.run([str(ROOT / "scripts/serve.sh"), "profile", "--gpu-memory-utilization", "0.85",
                    "--profiler-config", json.dumps(CFG)], cwd=ROOT, check=True, env={**os.environ, **env})
    for c, wait in CONC.items():
        sweep.wait_quiet()
        n = 8 if c == 1 else 6 * c  # long enough that the client outlives the profiling window
        tmp = OUT / f"client-c{c}.jsonl"
        cl = subprocess.Popen(["taskset", "-c", sweep.CLIENT_CPUS, sys.executable, "-m", "sl.client",
                               "--workload", "W1", "--n", str(n), "--seed", "0", "--concurrency", str(c),
                               "--out", str(tmp)], cwd=ROOT)
        time.sleep(wait)
        before = set(glob.glob(str(TRACES / "**" / "*.json*"), recursive=True))
        post("/start_profile", {"profile_prefix": f"c{c}", "max_iterations": PROFILE_ITERS})
        time.sleep(5)
        post("/stop_profile")
        cl.wait()
        new = set(glob.glob(str(TRACES / "**" / "*.json*"), recursive=True)) - before
        print(f"c={c}: new trace files {sorted(new)}")
        time.sleep(5)
    subprocess.run(["docker", "logs", "sl-vllm"], stdout=(OUT / "server.log").open("w"),
                   stderr=subprocess.STDOUT)
    subprocess.run(["docker", "rm", "-f", "sl-vllm"], capture_output=True)


def classify(name: str) -> str:
    n = name.lower()
    if "flash" in n or "attn" in n or "attention" in n or "paged" in n:
        return "attention"
    if "gemm" in n or "gemv" in n or "cutlass" in n or "sm80_xmma" in n or "ampere" in n or "matmul" in n:
        return "gemm"
    return "other"


def union(iv: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0.0)


def analyze_file(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    kern = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    # Engine iterations: vLLM's per-step record_function ranges on the CPU side.
    steps = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("user_annotation", "cpu_op")
             and str(e.get("name", "")).startswith(("execute_model", "gpu_model_runner"))]
    t0 = min(k["ts"] for k in kern)
    t1 = max(k["ts"] + k["dur"] for k in kern)
    by = defaultdict(float)
    names = defaultdict(float)
    for k in kern:
        by[classify(k["name"])] += k["dur"]
        names[k["name"][:90]] += k["dur"]
    return {"file": Path(path).name, "kernels": len(kern), "span_us": t1 - t0,
            "busy_us": union([(k["ts"], k["ts"] + k["dur"]) for k in kern]),
            "split_us": dict(by), "step_ranges": len(steps),
            "step_names": sorted({s["name"] for s in steps})[:5],
            "top": sorted(names.items(), key=lambda x: -x[1])[:12]}


def analyze() -> None:
    res = {}
    for f in sorted(glob.glob(str(TRACES / "**" / "*.json*"), recursive=True)):
        res[f] = analyze_file(f)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1))
    for f, r in res.items():
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
