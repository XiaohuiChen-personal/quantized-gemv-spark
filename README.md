# Trading Bits for Bandwidth

Quantized GEMV kernels for LLM decoding on the DGX Spark.

Xiaohui Chen · Course project, EN.605.617.81 Introduction to GPU Programming · Fall 2026

When a language model generates text one token at a time, every linear layer is a
matrix-vector product **y = W·x** (GEMV), and every weight is read from memory once per
token for a single multiply-add. The kernel is limited by memory bandwidth, so its time is
roughly the bytes read divided by the bandwidth it reaches. This project builds a ladder of
hand-written CUDA kernels that pull two levers: reach the bandwidth limit with BF16
weights, then read fewer bytes by storing the weights as INT8 or INT4. Each kernel is
checked against a CPU reference and timed on real Qwen3.5-4B weight matrices on an NVIDIA
GB10 (DGX Spark), to test whether the speedups track the byte counts.

## Status

Kernels, harness, tests and data scripts in place; measurements in progress (see `results/`).

## Layout (planned)

| Path | Role |
| --- | --- |
| `Makefile` | Builds every program with one `make` |
| `cuda_check.h` | `CUDA_CHECK` wrapper for every CUDA call |
| `src/` | Kernels, benchmark harness, CPU reference |
| `scripts/` | Weight export and chart scripts |
| `data/` | Exported weight matrices (`.bin`, not committed) and their shape manifest |
| `results/` | Kept benchmark output and run logs |
