"""Workloads from DESIGN.md: exact prompt token IDs and output lengths, fixed by seed."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B and 1.5B share this tokenizer


@dataclass(frozen=True)
class Request:
    idx: int
    prompt: list[int]
    max_tokens: int
    kind: str  # "short", "long" or "prefix"


@lru_cache(maxsize=1)
def token_stream() -> np.ndarray:
    """All GSM8K train questions, tokenized once into one ID stream."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    path = hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset")
    questions = pq.read_table(path, columns=["question"]).column("question").to_pylist()
    ids: list[int] = []
    for enc in tok.encode_batch([q + "\n\n" for q in questions]):
        ids.extend(enc.ids)
    return np.asarray(ids, dtype=np.int64)


def _lognormal(rng: np.random.Generator, median: float, sigma: float, lo: int, hi: int) -> int:
    return int(np.clip(round(float(rng.lognormal(np.log(median), sigma))), lo, hi))


def _slice(rng: np.random.Generator, stream: np.ndarray, n: int) -> list[int]:
    start = int(rng.integers(0, len(stream) - n))
    return stream[start:start + n].tolist()


def _short(rng, stream, i):
    return Request(i, _slice(rng, stream, _lognormal(rng, 200, 0.8, 32, 1024)),
                   _lognormal(rng, 200, 0.8, 16, 1024), "short")


def _long(rng, stream, i):
    return Request(i, _slice(rng, stream, _lognormal(rng, 2000, 0.6, 512, 6000)),
                   _lognormal(rng, 100, 0.6, 16, 512), "long")


def build(workload: str, n: int, seed: int) -> list[Request]:
    stream = token_stream()
    rng = np.random.default_rng(seed)
    if workload == "W1":
        return [_short(rng, stream, i) for i in range(n)]
    if workload == "W2":
        return [_long(rng, stream, i) for i in range(n)]
    if workload == "Wmix":
        return [(_long if rng.random() < 0.2 else _short)(rng, stream, i) for i in range(n)]
    if workload == "W3":
        # Prefixes are fixed by a separate seed so every arrival seed shares the same 8 prefixes.
        prng = np.random.default_rng(10_000)
        prefixes = [_slice(prng, stream, 2000) for _ in range(8)]
        return [Request(i, prefixes[int(rng.integers(8))] + _slice(rng, stream, 100), 100, "prefix")
                for i in range(n)]
    raise ValueError(f"unknown workload {workload!r}")


def arrivals(n: int, rate: float, seed: int) -> np.ndarray:
    """Poisson arrival times (seconds from start) for n requests at `rate` req/s."""
    rng = np.random.default_rng(seed + 1_000_000)
    return np.cumsum(rng.exponential(1.0 / rate, size=n))
