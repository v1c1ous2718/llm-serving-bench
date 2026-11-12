"""Mock /v1/completions server that streams `max_tokens` chunks with no model behind it.

Used only for DESIGN.md check 2: how fast can the pinned client drive requests before
the client itself becomes the bottleneck?

uv run python -m sl.mock_server --port 8001 --token-delay-ms 0
"""
from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import web


def make_app(token_delay: float) -> web.Application:
    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        n = int(body["max_tokens"])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        chunk = {"choices": [{"index": 0, "text": "x", "finish_reason": None}]}
        line = b"data: " + json.dumps(chunk).encode() + b"\n\n"
        for _ in range(n):
            if token_delay:
                await asyncio.sleep(token_delay)
            await resp.write(line)
        usage = {"choices": [], "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": n}}
        await resp.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--token-delay-ms", type=float, default=0.0)
    a = ap.parse_args()
    web.run_app(make_app(a.token_delay_ms / 1000), port=a.port, access_log=None)


if __name__ == "__main__":
    main()
