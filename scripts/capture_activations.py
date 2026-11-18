#!/usr/bin/env python3
"""Record the real input vectors of every large GEMV of Qwen3.5-4B (runs in a container).

Runs the BF16 model (Hugging Face transformers) over consecutive windows of the WikiText-2
test text (scripts/text_data.py) and records, at every token position, the exact BF16
vector each large linear layer multiplies -- the x of y = W x in a decode step. A causal
model computes the same per-position inputs in one forward pass over a window as it would
token by token, so each position is one decode-step input.

Inputs shared by several matrices are stored once per layer ("site"):

  mlp_in       gate_proj and up_proj (post-attention RMSNorm output)
  down_in      down_proj (SwiGLU output, K = 9216)
  attn_in      DeltaNet in_proj_qkv/in_proj_z, or attention q/k/v_proj (input RMSNorm output)
  attn_out_in  DeltaNet out_proj, or attention o_proj (K = 4096)
  lm_in        the LM head (final RMSNorm output)

Output (data/real_x/, not committed): <layer>.<site>.bf16.bin, row-major [tokens, K] BF16,
and manifest.json (shapes, SHA-256, the windows, the model's loss on each window as a
sanity check, versions). The shared-input assumption is checked bit for bit.

Usage (see scripts/run_stage.sh realx):
  python3 scripts/capture_activations.py [--windows 4] [--len 512] [--out data/real_x]
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_data  # noqa: E402

SITE = {"gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in",
        "in_proj_qkv": "attn_in", "in_proj_z": "attn_in", "q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in",
        "out_proj": "attn_out_in", "o_proj": "attn_out_in"}
NAME_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(mlp|linear_attn|self_attn)\.(\w+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--out", default="data/real_x")
    a = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    snap = text_data.model_snapshot()
    tok = AutoTokenizer.from_pretrained(snap)
    ids = text_data.wikitext_ids(tok)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(snap, dtype=torch.bfloat16).cuda().eval()
    print(f"model loaded in {time.time() - t0:.0f} s; {len(ids)} WikiText-2 test tokens", flush=True)

    store = defaultdict(list)       # (layer, site) -> list of [L, K] uint16 arrays, one per window
    shared = {}                     # (layer, site) -> first module's input of the current window
    mismatches = []

    def hook_for(layer, site, mod_name):
        def hook(_m, args):
            x = args[0][0].detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            key = (layer, site)
            if key in shared:
                if not np.array_equal(shared[key], x):
                    mismatches.append(mod_name)
            else:
                shared[key] = x
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        m = NAME_RE.match(name)
        if m and m.group(3) in SITE:
            hooks.append(mod.register_forward_pre_hook(hook_for(int(m.group(1)), SITE[m.group(3)], name)))
        elif name == "lm_head":
            hooks.append(mod.register_forward_pre_hook(hook_for(-1, "lm_in", name)))
    if len(hooks) != 201:  # 24 DeltaNet layers x 6 + 8 attention layers x 7 + the LM head
        raise SystemExit(f"expected 201 large linear layers, hooked {len(hooks)}")
    windows, losses = [], []
    with torch.no_grad():
        for w in range(a.windows):
            lo = w * a.len
            inp = torch.tensor([ids[lo:lo + a.len]], device="cuda")
            shared.clear()
            out = model(input_ids=inp, labels=inp)
            if len(shared) != 32 * 4 + 1:
                raise SystemExit(f"window {w}: {len(shared)} sites recorded, expected {32 * 4 + 1}")
            for k, v in shared.items():
                store[k].append(v)
            windows.append([lo, lo + a.len])
            losses.append(float(out.loss))
            print(f"window {w}: tokens {lo}..{lo + a.len}, loss {out.loss:.4f} (ppl {math.exp(out.loss):.2f}), "
                  f"{len(shared)} sites [{time.time() - t0:.0f} s]", flush=True)
    for h in hooks:
        h.remove()
    if mismatches:
        raise SystemExit(f"modules sharing a site saw different inputs: {sorted(set(mismatches))[:5]}")
    os.makedirs(a.out, exist_ok=True)
    entries = []
    for (layer, site), parts in sorted(store.items()):
        arr = np.concatenate(parts, axis=0)
        fname = f"{'lm' if layer < 0 else f'L{layer}'}.{site}.bf16.bin"
        path = os.path.join(a.out, fname)
        arr.tofile(path)
        entries.append({"layer": layer, "site": site, "tokens": arr.shape[0], "K": arr.shape[1], "file": fname,
                        "sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
    man = {"model": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
           "text": json.load(open("data/extras.json"))["dataset"], "windows": windows, "window_len": a.len,
           "loss_per_window": losses, "ppl_all_windows": math.exp(sum(losses) / len(losses)),
           "transformers": transformers.__version__, "torch": torch.__version__, "git": text_data.git_desc(),
           "shared_inputs_identical": True, "sites": entries}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{len(entries)} sites x {a.windows * a.len} tokens written to {a.out} "
          f"in {time.time() - t0:.0f} s; perplexity over the windows {man['ppl_all_windows']:.2f}")


if __name__ == "__main__":
    main()
