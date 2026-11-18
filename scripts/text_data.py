"""The text the model-level experiments run on: the WikiText-2 (raw) test split, pinned
to the dataset revision recorded in data/extras.json by scripts/fetch_extras.py.

The split is joined with blank lines, as in the Hugging Face perplexity guide, and
tokenized once without special tokens; experiments take consecutive windows from the
start of that token stream, so every run (activation capture, decode prompt, perplexity)
uses the same tokens.
"""
import hashlib
import json
import os


def wikitext_ids(tokenizer, extras="data/extras.json"):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    e = json.load(open(extras))["dataset"]
    p = hf_hub_download(e["repo"], e["file"], repo_type="dataset", revision=e["revision"], local_files_only=True)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != e["sha256"]:
        raise SystemExit(f"{p}: sha256 {h} != {e['sha256']} recorded in {extras}")
    text = "\n\n".join(pq.read_table(p).column("text").to_pylist())
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def model_snapshot():
    """The pinned Qwen3.5-4B snapshot in the local Hugging Face cache."""
    from huggingface_hub import snapshot_download

    return snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                             local_files_only=True)


def git_desc():
    """Inside a container the mounted repository belongs to another uid, so git refuses
    to describe it; run_stage.sh passes the description in GIT_DESC."""
    return os.environ.get("GIT_DESC", "unknown")
