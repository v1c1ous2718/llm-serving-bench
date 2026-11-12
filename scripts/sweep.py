"""Run every configuration in DESIGN.md, one server session at a time. Resumable: finished files are skipped.

  sg docker -c "uv run python scripts/sweep.py"            # everything
  sg docker -c "uv run python scripts/sweep.py baseline F1-seqs8"

Per session: start the server (scripts/serve.sh), batch-1 drift point, closed-loop capacity point for each
workload, the open-loop sweep (rates fixed from the *baseline* capacity, so every factor value is tested at
the same absolute rates), then the batch-1 drift point again. /metrics is sampled at 1 Hz during every point.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sl.summarize import load, summarize  # noqa: E402

RUNS = ROOT / "runs"
MULTIPLES = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25]
SEEDS = [0, 1, 2]
N = 600
CLIENT_CPUS = "10,11"
CAP_CONCURRENCY = 256
DRIFT_LIMIT = 0.05
QUIET_S = 10
MEM_LIMIT_MIB = 11_264  # vLLM at 0.75 holds ~9.2 GiB; the idle Windows desktop ~1 GiB

# name -> (server args, workloads, env). The baseline value of each factor is not repeated.
SESSIONS: dict[str, tuple[list[str], list[str], dict]] = {
    "baseline": ([], ["W1", "W2", "Wmix", "W3"], {}),
    "F1-seqs8": (["--max-num-seqs", "8"], ["W1"], {}),
    "F1-seqs32": (["--max-num-seqs", "32"], ["W1"], {}),
    "F1-seqs128": (["--max-num-seqs", "128"], ["W1"], {}),
    "F2-off": (["--no-enable-chunked-prefill"], ["Wmix"], {}),
    "F2-chunk512": (["--max-num-batched-tokens", "512"], ["Wmix"], {}),
    "F2-chunk8192": (["--max-num-batched-tokens", "8192"], ["Wmix"], {}),
    "F3-off": (["--no-enable-prefix-caching"], ["W3"], {}),
    "F4-kv16k": (["--num-gpu-blocks-override", "1000"], ["W1", "W2"], {}),
    "F4-kv64k": (["--num-gpu-blocks-override", "4000"], ["W1", "W2"], {}),
    "F5-1.5B": ([], ["W1"], {"MODEL": "Qwen/Qwen2.5-1.5B-Instruct"}),
}
METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
           "vllm:num_preemptions_total")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (RUNS / "sweep.log").open("a") as f:
        f.write(line + "\n")


def gpu_state() -> dict:
    """Whole-card state from nvidia-smi; it also sees Windows-side use that WSL cannot list as processes."""
    q = "memory.used,utilization.decoder,utilization.encoder,clocks.sm,clocks.mem,power.draw"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    keys = ["mem_used_mib", "dec_pct", "enc_pct", "sm_mhz", "mem_mhz", "power_w"]
    try:
        return dict(zip(keys, (float(v) for v in out.split(", "))))
    except ValueError:
        return {}


def wait_quiet() -> None:
    """Block until nothing outside vLLM uses the card: no video decode/encode, total memory under the limit,
    for QUIET_S consecutive seconds. A video player on the Windows side made batch-1 decode 5x slower."""
    quiet, warned = 0, False
    while quiet < QUIET_S:
        g = gpu_state()
        ok = bool(g) and g["dec_pct"] == 0 and g["enc_pct"] == 0 and g["mem_used_mib"] < MEM_LIMIT_MIB
        quiet = quiet + 1 if ok else 0
        if not ok and not warned:
            log(f"waiting for a quiet GPU: {g}")
            warned = True
        time.sleep(1)
    if warned:
        log("GPU quiet again, continuing")


def sampler(path: Path, stop: threading.Event) -> None:
    with path.open("w") as f:
        while not stop.is_set():
            try:
                text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=2).read().decode()
                row = {"t": time.time()}
                for line in text.splitlines():
                    name = line.split("{", 1)[0]
                    if name in METRICS:
                        row[name.removeprefix("vllm:")] = float(line.rsplit(" ", 1)[1])
                row.update(gpu_state())
                f.write(json.dumps(row) + "\n")
            except OSError:
                pass
            stop.wait(1.0)


def point(out: Path, workload: str, seed: int, n: int, mode: str, value: float) -> Path:
    if out.exists():
        return out
    wait_quiet()
    tmp = out.with_suffix(".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(out.with_suffix(".metrics.jsonl"), stop), daemon=True)
    th.start()
    cmd = ["taskset", "-c", CLIENT_CPUS, sys.executable, "-m", "sl.client", "--workload", workload,
           "--n", str(n), "--seed", str(seed), f"--{mode}", str(value), "--out", str(tmp)]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        stop.set()
        th.join()
    tmp.rename(out)
    time.sleep(5)
    return out


def drift(session: str, tag: str) -> float:
    out = point(RUNS / session / tag / "W1" / "0.jsonl", "W1", 0, 60, "concurrency", 1)
    return summarize(load(out, 10))["tpot_ms"]["p50"]


def capacity(session: str, workload: str) -> float:
    out = point(RUNS / session / "capacity" / workload / "0.jsonl", workload, 0, N, "concurrency",
                CAP_CONCURRENCY)
    return summarize(load(out, 50))["req_per_s"]


def done(session: str) -> bool:
    return (RUNS / session / "bs1_end" / "W1" / "0.jsonl").exists()


def run_session(name: str, base_cap: dict[str, float]) -> None:
    args, workloads, env = SESSIONS[name]
    if done(name):
        log(f"{name}: already complete")
        return
    log(f"{name}: starting server {args} {env}")
    subprocess.run([str(ROOT / "scripts/serve.sh"), name, *args], cwd=ROOT, check=True,
                   env={**os.environ, **env})
    start = drift(name, "bs1")
    log(f"{name}: batch-1 TPOT p50 {start:.2f} ms")
    for w in workloads:
        c = capacity(name, w)
        log(f"{name}: {w} closed-loop capacity {c:.2f} req/s")
        if name == "baseline":
            base_cap[w] = c
            (RUNS / "capacity.json").write_text(json.dumps(base_cap, indent=1))
        for seed in SEEDS:
            for m in MULTIPLES:
                rate = round(m * base_cap[w], 3)
                out = point(RUNS / name / f"x{m}" / w / f"{seed}.jsonl", w, seed, N, "rate", rate)
                s = summarize(load(out, 50))
                log(f"{name}: {w} x{m} ({rate} req/s) seed {seed}: TTFT p95 {s['ttft_s']['p95']:.3f} s, "
                    f"TPOT p95 {s['tpot_ms']['p95']:.1f} ms, errors {s['errors']}")
    end = drift(name, "bs1_end")
    d = end / start - 1
    verdict = "valid" if abs(d) <= DRIFT_LIMIT else "INVALID"
    log(f"{name}: drift {start:.2f} -> {end:.2f} ms ({d:+.1%}), {verdict}")
    with (RUNS / "sessions.jsonl").open("a") as f:
        f.write(json.dumps({"session": name, "args": args, "env": env, "bs1_start_ms": start,
                            "bs1_end_ms": end, "drift": d, "valid": verdict == "valid"}) + "\n")


def main() -> None:
    names = sys.argv[1:] or list(SESSIONS)
    cap_file = RUNS / "capacity.json"
    base_cap = json.loads(cap_file.read_text()) if cap_file.exists() else {}
    if "baseline" not in names and not base_cap:
        sys.exit("baseline capacities missing; run the baseline session first")
    for name in names:
        try:
            run_session(name, base_cap)
        except Exception as e:  # one broken session must not stop the night run
            log(f"{name}: FAILED ({e}); continuing with the next session")
    subprocess.run(["docker", "rm", "-f", "sl-vllm"], capture_output=True)
    log("sweep finished")


if __name__ == "__main__":
    main()
