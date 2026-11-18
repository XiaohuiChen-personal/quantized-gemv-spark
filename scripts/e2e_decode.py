#!/usr/bin/env python3
"""End-to-end decode speed and perplexity of one Qwen3.5-4B checkpoint in vLLM (container).

Batch 1, greedy, one request at a time, prefix caching off, CUDA graphs on (vLLM's
default), the prompt given as token ids (the first P tokens of the WikiText-2 test text,
scripts/text_data.py). Decode speed by difference, so the fixed per-request costs
(scheduling, prefill, the first token) cancel:

    decode tokens/s = D / (median t(max_tokens = 1 + D) - median t(max_tokens = 1))

with the two lengths alternated over R repetitions after a warm-up, for each prompt
length P. Every generated token is forced (ignore_eos), so all runs decode the same
number of steps. Also recorded: the greedy continuation (to compare formats) and the
perplexity of the WikiText-2 test text, from prompt log-probabilities over N windows of
W tokens (the first token of each window is not scored).

Usage: python3 scripts/e2e_decode.py --model DIR --label bf16|int8|int4 --out results/e2e/x.json
           [--contexts 128,2048] [--decode 256] [--reps 10] [--ppl-windows 8] [--ppl-len 1024]
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="128,2048")
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ppl-windows", type=int, default=8)
    ap.add_argument("--ppl-len", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    a = ap.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = text_data.wikitext_ids(tok)
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=False, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    load_s = time.time() - t0
    quant = getattr(llm.llm_engine.vllm_config.model_config, "quantization", None)
    print(f"[{a.label}] loaded in {load_s:.0f} s; quantization={quant}", flush=True)

    def gen(prompt, n):
        sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True, detokenize=False)
        torch.cuda.synchronize()
        s = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        e = time.perf_counter()
        toks = list(out[0].outputs[0].token_ids)
        if len(toks) != n:
            raise SystemExit(f"asked for {n} tokens, got {len(toks)}")
        return e - s, toks

    result = {"label": a.label, "model": a.model, "quantization": quant, "git": text_data.git_desc(),
              "vllm": vllm.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_s": load_s,
              "decode_tokens": a.decode, "reps": a.reps, "contexts": {}}
    for P in [int(c) for c in a.contexts.split(",")]:
        prompt = TokensPrompt(prompt_token_ids=ids[:P])
        for _ in range(a.warmup):
            gen(prompt, 1 + a.decode)
            gen(prompt, 1)
        short, long_, cont = [], [], None
        for r in range(a.reps):
            ts, _ = gen(prompt, 1)
            tl, toks = gen(prompt, 1 + a.decode)
            short.append(ts)
            long_.append(tl)
            if cont is None:
                cont = toks
            elif toks != cont:
                print(f"[{a.label}] P={P} rep {r}: greedy continuation changed between repetitions", flush=True)
        dt = st.median(long_) - st.median(short)
        c = {"t_first_s": short, "t_full_s": long_, "decode_tokens_per_s": a.decode / dt,
             "ms_per_token": 1e3 * dt / a.decode, "prefill_plus_first_ms": 1e3 * st.median(short),
             "pairwise_tokens_per_s": [a.decode / (l - s) for s, l in zip(short, long_)], "continuation": cont}
        result["contexts"][str(P)] = c
        print(f"[{a.label}] P={P}: {c['decode_tokens_per_s']:.2f} decode tokens/s ({c['ms_per_token']:.3f} ms/token), "
              f"first token {c['prefill_plus_first_ms']:.1f} ms; pairwise {min(c['pairwise_tokens_per_s']):.2f}–"
              f"{max(c['pairwise_tokens_per_s']):.2f}", flush=True)
    result["continuation_text"] = tok.decode(result["contexts"][str(min(int(c) for c in a.contexts.split(",")))]["continuation"][:64])

    if a.ppl_windows:
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
        nll, n = [], 0
        for w in range(a.ppl_windows):
            win = ids[w * a.ppl_len:(w + 1) * a.ppl_len]
            out = llm.generate([TokensPrompt(prompt_token_ids=win)], sp, use_tqdm=False)[0]
            lp = out.prompt_logprobs
            s = -sum(lp[i][win[i]].logprob for i in range(1, len(win)))
            nll.append(s)
            n += len(win) - 1
        result["ppl"] = {"windows": a.ppl_windows, "window_len": a.ppl_len, "nll_per_window": nll,
                         "perplexity": math.exp(sum(nll) / n)}
        print(f"[{a.label}] WikiText-2 perplexity {result['ppl']['perplexity']:.4f} over {n} scored tokens", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
