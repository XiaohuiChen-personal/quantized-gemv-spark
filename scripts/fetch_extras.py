#!/usr/bin/env python3
"""Fetch what running the model needs beyond the weight matrices (host Python).

1. The tokenizer, chat template and processor files of Qwen3.5-4B at the pinned revision
   (the weight export only needed the safetensors files and their index); they go into
   the same Hugging Face cache snapshot as the weights.
2. The WikiText-2 (raw) test split (Salesforce/wikitext, parquet), the text whose real
   activations scripts/capture_activations.py records and that the end-to-end runs use
   as prompt and perplexity text. The split is joined with blank lines (as in the
   Hugging Face perplexity guide) into data/wikitext2_test.txt, so the containers need no
   parquet reader.

Writes data/extras.json with the dataset revision and SHA-256 of every file.

Usage: python3 scripts/fetch_extras.py [--data data]
"""
import argparse
import hashlib
import json
import os
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_weights as ew  # noqa: E402

MODEL_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json", "video_preprocessor_config.json"]
DATASET = "Salesforce/wikitext"
DATASET_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # the dataset's head on 2026-11-18, pinned


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    out = {"model": f"{ew.REPO}@{ew.REVISION}", "model_files": {}}
    for f in MODEL_FILES:
        p = hf_hub_download(ew.REPO, f, revision=ew.REVISION)
        out["model_files"][f] = sha256(p)
    rev = DATASET_REVISION
    p = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=rev)
    import pyarrow.parquet as pq
    txt = os.path.join(a.data, "wikitext2_test.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pq.read_table(p).column("text").to_pylist()))
    out["dataset"] = {"repo": DATASET, "revision": rev, "file": DATASET_FILE, "sha256": sha256(p),
                      "text_file": "data/wikitext2_test.txt", "text_sha256": sha256(txt)}
    with open(os.path.join(a.data, "extras.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
