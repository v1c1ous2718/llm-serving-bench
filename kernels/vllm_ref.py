"""vLLM reference under bench.py's conditions (DESIGN.md variant V). kernels/run.sh vllm_ref.py [FLASH_ATTN|TRITON_ATTN]
Offline LLM.generate, same 512-token prompt, greedy, ignore_eos. Decode ms/token = (t(257 tokens) - t(1 token)) / 256,
median of 3, for batch 1 and 8 (8 identical prompts; prefix caching serves the prefill in both terms).
"""
import json
import os
import statistics
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from engine import find_model


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "FLASH_ATTN"
    path = find_model()
    ids = AutoTokenizer.from_pretrained(path)(open(f"{path}/README.md").read())["input_ids"][:512]
    llm = LLM(model=path, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.6,
              attention_backend=backend, seed=0)


    def t(n, B):
        sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
        prompts = [{"prompt_token_ids": ids}] * B
        s = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - s


    rows = []
    for B in (1, 8):
        t(257, B)                                                         # warm-up
        t1 = statistics.median(t(1, B) for _ in range(3))
        tn = statistics.median(t(257, B) for _ in range(3))
        rows.append({"variant": f"V-{backend}", "batch": B, "e2e_ms": round((tn - t1) / 256 * 1e3, 3),
                     "date": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "vLLM 0.32.0 offline LLM.generate"})
    os.makedirs("results", exist_ok=True)
    with open("results/bench.jsonl", "a") as f:
        for r in rows:
            print(json.dumps(r), flush=True)
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
