"""Load generator for an OpenAI-compatible /v1/completions endpoint (DESIGN.md).

Open loop:   uv run python -m sl.client --workload W1 --rate 4 --n 600 --seed 0 --out runs/x.jsonl
Closed loop: uv run python -m sl.client --workload W1 --concurrency 64 --n 600 --seed 0 --out runs/x.jsonl

Every record holds raw perf_counter timestamps relative to the run start; analysis happens elsewhere.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from sl import workloads


async def send(session: aiohttp.ClientSession, url: str, model: str, req: workloads.Request,
               t0: float, scheduled: float | None) -> dict:
    body = {
        "model": model, "prompt": req.prompt, "max_tokens": req.max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = {"idx": req.idx, "kind": req.kind, "prompt_len": len(req.prompt),
           "max_tokens": req.max_tokens, "scheduled": scheduled, "chunks": [], "error": None}
    rec["send"] = time.perf_counter() - t0
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return rec
            buf = b""
            async for data in resp.content.iter_any():
                now = time.perf_counter() - t0
                buf += data
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    line = event.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    msg = json.loads(payload)
                    if msg.get("usage"):
                        rec["prompt_tokens"] = msg["usage"]["prompt_tokens"]
                        rec["output_tokens"] = msg["usage"]["completion_tokens"]
                    if msg.get("choices") and msg["choices"][0].get("text"):
                        rec["chunks"].append(now)
    except Exception as e:  # noqa: BLE001 - every failure is recorded, not raised
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["end"] = time.perf_counter() - t0
    return rec


async def run_open(url, model, reqs, times, out) -> None:
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = []
        for req, at in zip(reqs, times):
            delay = at - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(session, url, model, req, t0, float(at))))
        write(out, await asyncio.gather(*tasks))


async def run_closed(url, model, reqs, concurrency, out) -> None:
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    queue: asyncio.Queue = asyncio.Queue()
    for r in reqs:
        queue.put_nowait(r)
    results: list[dict] = []
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t0 = time.perf_counter()

        async def worker():
            while not queue.empty():
                results.append(await send(session, url, model, queue.get_nowait(), t0, None))

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    write(out, sorted(results, key=lambda r: r["idx"]))


def write(out: Path, records: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    errors = sum(r["error"] is not None for r in records)
    span = max(r["end"] for r in records) - min(r["send"] for r in records)
    print(f"{out}: {len(records)} requests, {errors} errors, {span:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--workload", required=True, choices=["W1", "W2", "Wmix", "W3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rate", type=float, help="open loop, Poisson arrivals (req/s)")
    mode.add_argument("--concurrency", type=int, help="closed loop, fixed in-flight requests")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    reqs = workloads.build(a.workload, a.n, a.seed)
    if a.rate:
        asyncio.run(run_open(a.url, a.model, reqs, workloads.arrivals(a.n, a.rate, a.seed), a.out))
    else:
        asyncio.run(run_closed(a.url, a.model, reqs, a.concurrency, a.out))


if __name__ == "__main__":
    main()
