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

Results and hypothesis verdicts: [`results/ANALYSIS.md`](results/ANALYSIS.md) (generated
from the kept CSVs by `scripts/analyze.py`), figures in [`results/figures/`](results/figures/).
A guide for code reviewers is in [`REVIEW.md`](REVIEW.md).

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA GB10 (DGX Spark), compute capability 12.1, 48 SMs, 24 MiB L2, 128 GB unified LPDDR5x (273 GB/s rated) |
| OS / host compiler | Ubuntu 24.04 (aarch64), gcc 11.5 |
| CUDA | 13.0 (`nvcc` V13.0.88), driver 580.159.03, cuBLAS / cuBLASLt from the toolkit |
| Python (data, analysis) | 3.12 with NumPy, pandas, matplotlib; `huggingface_hub` for the download |
| Model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (Apache-2.0) |

Every number in `results/` was produced on this machine; each CSV starts with a
provenance line (UTC time, host, device, driver, CUDA and cuBLAS versions, git commit).

## Kernels

All kernels compute y = W·x for one BF16 input vector x (batch 1), W row-major M × K,
FP32 accumulation, BF16 output y.

| Kernel | Weights | Design |
| --- | --- | --- |
| `v0` | BF16 | one thread per row (uncoalesced; the naive start) |
| `v1` | BF16 | one warp per row, 2-byte loads, warp-shuffle reduction |
| `v2` | BF16 | 16-byte loads (8 weights), x staged in shared memory, persistent grid sized by the occupancy API, several rows per warp and several loads in flight per lane |
| `v3` | INT8, one FP16 scale per row | v2's structure; 16 weights per load, decoded in registers, the row scale applied once |
| `v4` | INT4, one FP16 scale per 128 weights along K | v2's structure; 32 weights per load, one scale per load |
| `cublas` | BF16 | `cublasGemmEx(CUBLAS_OP_T, …, M, 1, K, …, CUBLAS_COMPUTE_32F)` on the row-major matrix |
| `cublaslt` | BF16 | cuBLASLt with each of its (up to 8) heuristic algorithms for the shape |

v2, v3 and v4 are one template (`gemv_vec` in `src/kernels.cuh`) parameterised by the
weight format, so they differ only in how a 16-byte chunk is decoded. Their
configurations (threads, rows per warp, unroll, x in shared memory, grid) were chosen by
a tuning run with a rule fixed beforehand and are frozen in `src/tuned.h`.
Diagnostics, not rungs: `v2abl` (v2 with one factor changed), `v4load` (v4's loads
without the decode), `read` (a plain read of the same bytes: the size-matched
bandwidth reference).

**Quantized formats** (`scripts/quantize.py`): symmetric, round-to-nearest (half to
even), no zero points. The scale (max|w| / 127 per row for INT8, max|w| / 7 per group of
128 for INT4) is rounded to FP16 first and q = rint(w / scale) uses that FP16 value.
INT4 codes are two's complement, two per byte, **low nibble = even column k, high nibble
= odd column k + 1**; scales are `[M][K/128]`. The kernels require K % 128 == 0 (every
benchmarked K is a multiple of 128); M is arbitrary.

## Build and test

```bash
make              # build/bench, build/test, build/bandwidth (ARCH=-arch=native by default;
                  # on a machine without a GPU pass e.g. ARCH=-arch=sm_121)
make test         # CUDA unit tests + Python quantizer tests
```

`build/test` runs every kernel configuration the benchmark can use (the ladder,
cuBLAS and all 216 tuning configurations; cuBLASLt's heuristic algorithms are only
checked, not required to pass, in the benchmark) on awkward shapes with random and ±1
inputs, exact one-hot checks (x = e_k must return bf16(w[:, k]) bit for bit, at chunk
and group edges, over all 16 INT4 codes), and mutation tests that must make the checker
fail (a changed weight, the last 32 columns zeroed, one column zeroed, nibbles swapped,
group scales swapped, the last row not computed). Expected ending:

```
kernels: 231 tested, 28655 checks passed, 0 failed so far
total: 28721 checks passed, 0 failed
ALL TESTS PASSED
```

## Data

```bash
python3 -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-4B', \
    revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', allow_patterns=['*.safetensors', '*.json'])"
make data         # export 11 matrices (raw BF16, ~1.6 GB), quantize them, write the x vectors
```

`scripts/export_weights.py` copies each tensor's bytes out of the safetensors files
(they are already row-major BF16) and writes `data/manifest.tsv`/`.json` (shapes,
SHA-256) and `data/inventory.csv` (every 2-D language-model weight, for the per-token
model). The `.bin` files are not committed. Benchmarked: layer-0 MLP `gate_proj`,
`up_proj`, `down_proj`; layer-0 DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; layer-3
attention `q_proj`, `k_proj`, `v_proj`, `o_proj`; the LM head (= `embed_tokens`,
248 320 × 2560).

## Run the measurements

```bash
scripts/run_stage.sh test       # -> results/tests/
scripts/run_stage.sh tune       # -> results/tuning/tune.csv; then
python3 scripts/select_tuned.py results/tuning/tune.csv   # writes src/tuned.h
scripts/run_stage.sh final      # bandwidth reference + 3 executions -> results/final/
scripts/run_stage.sh rotation   # rotation sweep with the frozen kernels -> results/rotation/
scripts/run_stage.sh nsys       # Nsight Systems cross-check -> results/nsys/
scripts/run_stage.sh sweep      # accuracy of every large matrix of all 32 layers (CPU)
python3 scripts/analyze.py      # verdicts, tables, figures -> results/ANALYSIS.md
```

One benchmark pass by hand: `./build/bench --mode final --out results/x.csv` (also
`--mode tune|quick|rotation`, `--only gate,lmhead`, `--seed N`, `--samples file.csv`).
Nothing else should use the GPU while timing; on the GB10 the CPU shares the memory bus,
so the scripts also refuse to run while another process holds the GPU.

## How the timing works

- **Rotation:** each launch reads a different device copy of the matrix; R = max(2,
  ⌈32 × L2 / stored bytes⌉) copies (an 805 MB footprint). The planned 4 × L2 was not
  enough on the GB10: the rotation sweep (`results/rotation/`) showed rates up to 14% above
  the DRAM plateau there. Even at 32 × L2 about one launch per rotation cycle runs faster
  than the rest; medians ignore it.
- **Isolated** (the measure every pre-registered verdict uses): 200 launches, each between
  its own pair of CUDA events on an otherwise idle GPU, synchronized after each launch.
  Events add a small overhead to the kernel's own duration (median 2.4 µs over the final
  set, against Nsight Systems kernel durations); for cuBLAS they also catch part of its
  host-side call path.
- **Streamed:** whole rotation cycles (at least 200 launches) captured in one CUDA graph and
  replayed 5 times; per-launch time = replay time / launches. This is how a decode engine
  runs kernels and is used for the per-token model and for the Marlin comparison.
- **Spin** (diagnostic only): isolated with a spin kernel before the start event. It was
  briefly the isolated measure; on the GB10 it added overhead and quantized the results
  (`results/timing_method/`), so it is only recorded.
- Warm-up: at least 20 launches and 200 ms. Kernel order is shuffled in each of the three
  final executions. GPU clocks, power and temperature are logged next to every run.
- **Correctness before and after timing**, against a double-precision CPU reference
  built from the same inputs: every element within 2⁻⁸·|y| + 2⁻¹⁴·Σ|w·x| and a relative
  L2 error ≤ 4 × 10⁻³, for a Gaussian x and a ±1 x; the output after timing must also be
  bitwise identical to the first.
- **GB/s** counts weights + scales + x + y; GB = 10⁹ bytes.

## Layout

| Path | Role |
| --- | --- |
| `Makefile` | Builds every program with one `make`; `make test`, `make data` |
| `src/kernels.cuh` | The kernels (v0, v1, the `gemv_vec` template and its formats, read and spin kernels) |
| `src/gemv.cuh` | Launchers, rotating copies, timing, output checks |
| `src/kernel_sets.cuh`, `src/tuned.h` | Tuning grid, final kernel set and ablation; frozen configurations |
| `src/reference.h` | CPU reference in double precision; independent check of the quantizer's output |
| `src/bench.cu`, `src/test.cu`, `src/bandwidth.cu` | Benchmark driver, unit tests, 1 GiB bandwidth reference |
| `src/cuda_check.h`, `src/host_util.h` | `CUDA_CHECK`/`CUBLAS_CHECK`; BF16/FP16 conversions, file loading |
| `scripts/` | Export, quantization, inputs, tuning selection, pipeline, accuracy sweep, Nsight cross-check, analysis |
| `data/` | Manifest and inventory (committed); exported matrices (`.bin`, not committed) |
| `results/` | Kept outputs (CSV, logs), `ANALYSIS.md`, `figures/` |
