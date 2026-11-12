# Pre-run checks (DESIGN.md "Checks before trusting any number")

## Check 2: client ceiling — done 2026-11-11

Mock server (`sl/mock_server.py`) pinned to CPUs 0–7, streaming each request's `max_tokens` chunks with a
5 ms sleep between them; client pinned to CPUs 10–11. Workload W1 (median 190 output tokens), open loop.
Raw files: `runs/_ceiling/`.

| Target rate (req/s) | Output chunks/s | Send lateness p99 | TTFT p99 | TPOT p50 / p99 |
|---:|---:|---:|---:|---:|
| 25 | 5,492 | 1.2 ms | 7.7 ms | 5.85 / 6.02 ms |
| 50 | 10,168 | 1.3 ms | 8.1 ms | 5.76 / 6.04 ms |
| 100 | 19,892 | 10.0 ms | 18.2 ms | 5.89 / 6.08 ms |
| 200 | 39,369 | 32.8 ms | 40.8 ms | 5.96 / 6.53 ms |

(Achieved req/s is below the target because the measured span includes the drain after the last arrival.)

- The ~0.85 ms above the 5 ms delay is constant from 25 to 100 req/s, so it comes from the mock server's
  sleep-and-write loop, not from load on the client.
- Acceptance rule, fixed now: client-induced error must stay under 2% of each SLO term,
  i.e. send lateness p99 < 20 ms (TTFT SLO 1 s) and TPOT p99 drift < 1 ms (TPOT SLO 50 ms).
- **Ceiling: 100 req/s, ~20k output tokens/s.** At 200 req/s lateness p99 (32.8 ms) breaks the rule.
- Consequence: any sweep point above 100 req/s, or above ~20k output tokens/s, must split arrivals across
  two client processes on separate CPUs. This is checked against each workload's capacity C after F0.

A zero-delay mock run is not informative for TPOT: the server writes all chunks before the client reads,
so they arrive in one TCP read (`runs/_ceiling/rate50_delay0.jsonl`, TPOT p50 0.004 ms).

## Check 1: GPU inside a container — passed 2026-11-11
First attempt failed (`could not select device driver ""`): the toolkit apt source held NVIDIA's 404 HTML page
instead of the repository line. After fixing the source: NVIDIA Container Toolkit 1.20.2, Docker runtimes
`nvidia runc`, and `docker run --gpus all ubuntu:24.04 nvidia-smi -L` lists the RTX 3060.
Host driver: KMD 621.14, CUDA UMD 13.5.

## Check 3: session drift
| Session | Batch-1 TPOT p50 at start | at end | Drift | Valid |
|---|---:|---:|---:|---|
| 2026-11-11 F0 | 4.58 ms | 4.54 ms | −0.9% | yes |

## Check 4: WSL2 pinned memory — warning present
v0.32.0 logs: `Pinned memory is not available on this platform; falling back to device memory for UVA buffers.`
Every result in this project is measured under this condition; it is listed as a known limit.
