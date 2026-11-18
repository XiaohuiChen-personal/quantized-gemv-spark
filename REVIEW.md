# Code review guide

Thank you for reviewing. This page says what the code does, where the parts that matter
are, and what I would most like checked. Build and run instructions are in the
[README](README.md); the results are in [`results/ANALYSIS.md`](results/ANALYSIS.md).

## What to look at first (about 600 lines)

1. **`src/kernels.cuh`** — the kernels. `gemv_v0` and `gemv_v1` are the textbook
   versions. `gemv_vec` is one template for v2 (BF16), v3 (INT8) and v4 (INT4); the
   format structs `FmtBF16`, `FmtI8`, `FmtI4` say how one 16-byte chunk of weights is
   loaded and decoded. Questions:
   - Is the INT4 decode in `FmtI4::dot8` right? Nibble n (from the least significant)
     of 32-bit word i of a chunk is weight 8i + n; low nibble = even column. The same
     order must hold in `scripts/quantize.py` (`pack_int4`), `src/reference.h`
     (`int4_at`) and `src/test.cu` (`Synth::set4`).
   - Every 32-weight chunk lies inside one 128-weight group, so v4 loads one scale per
     chunk (`c >> 2`). Is that indexing right for every K that is a multiple of 128?
   - Any out-of-bounds access for rows ≥ M (the `row0 + r < M` guards), for the scale
     loads, or for `x_smem`?
2. **`src/gemv.cuh`** — `time_launch` (how a launch is timed), `DeviceCopies` (rotating
   copies), `check_output` (pass/fail criteria). Questions:
   - Does the isolated measurement time only the kernel? (Each launch sits between its
     own pair of plain CUDA events and is synchronized before the next one. Against
     Nsight Systems kernel durations the events add a median 2.4 µs over the final set,
     which matters only for the smallest kernels. A variant with a spin kernel before
     the start event, meant to hide host launch latency, made the times worse on the
     GB10 (`results/timing_method/`), so it is recorded as a diagnostic only.)
   - Can two consecutive launches ever read the same copy?
   - Is the tolerance (2⁻⁸·|y_ref| + 2⁻¹⁴·Σ|w·x| per element, relative L2 ≤ 4 × 10⁻³)
     tight enough to catch a real bug and loose enough not to fail a correct kernel?
3. **`src/test.cu`** — the one-hot tests and the mutation tests. Would any plausible bug
   in `gemv_vec` slip past them?
4. **`scripts/quantize.py`** — symmetric round-to-nearest with FP16 scales. Is it
   correct for all-zero groups and for tiny groups whose scale underflows FP16?

## Design decisions you may disagree with

- **Isolated timing is the pre-registered measure**, with a CUDA-graph "streamed" mode
  next to it. Isolated is conservative for small kernels; streamed is closer to how a
  decode engine runs.
- **Rotation footprint 32 × L2** (805 MB per matrix and format), not the planned 4 × L2,
  because the rotation sweep in `results/rotation/` showed caching above 4 × L2 on this
  machine.
- **One tuned configuration per kernel** for all shapes, not one per shape.
- **cuBLAS gets the default `cublasGemmEx` algorithm** for the pre-registered
  comparison; cuBLASLt's heuristic algorithms are reported alongside.
- **Random Gaussian x** for accuracy, so the errors measure weight rounding, not model
  quality.

## What is out of scope

No attention kernel, no batching, no end-to-end model inference. Quantization is plain
round-to-nearest (no GPTQ/AWQ calibration).

## How to give feedback

File an issue or comment on the review thread with `file:line`. Bugs that change a
number in `results/` matter most; style comments are welcome too.
