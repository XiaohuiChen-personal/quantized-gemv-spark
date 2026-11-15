#!/usr/bin/env bash
# The measurement pipeline on the DGX Spark, one stage at a time:
#
#   scripts/run_stage.sh test      unit tests                        -> results/tests/
#   scripts/run_stage.sh tune      v2/v3/v4 configuration grid       -> results/tuning/tune.csv
#                                  (then: scripts/select_tuned.py, commit src/tuned.h)
#   scripts/run_stage.sh final     bandwidth reference + final set,  -> results/final/
#                                  three executions, shuffled order
#   scripts/run_stage.sh rotation  rotation sweep, frozen kernels    -> results/rotation/
#   scripts/run_stage.sh nsys      Nsight Systems trace of one pass  -> results/nsys/
#   scripts/run_stage.sh sweep     all-layer accuracy sweep (CPU)    -> results/accuracy/
#
# Every stage records provenance (UTC time, host, git commit, GPU state) and refuses to
# time anything while another process uses the GPU. Timed stages log GPU clocks, power
# and temperature once a second next to their results.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export OMP_WAIT_POLICY=PASSIVE  # OpenMP threads must not spin during GPU timing
cd "$(dirname "$0")/.."
stage=${1:?usage: run_stage.sh test|tune|final|rotation|nsys|sweep}

env_report() {
    echo "# $(date -u +%FT%TZ) host=$(hostname) git=$(git describe --always --dirty) stage=$stage"
    nvcc --version | grep release
    nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv
    echo "# other GPU compute processes (none expected):"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    echo "# load average: $(cat /proc/loadavg)"
}
require_quiet() {
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
        echo "another process is using the GPU; stop it before timing" >&2
        exit 1
    fi
}
GPULOG=
gpu_log_start() {
    nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
        --format=csv -lms 1000 > "$1" 2>&1 &
    GPULOG=$!
}
gpu_log_stop() { [ -n "$GPULOG" ] && kill "$GPULOG" 2>/dev/null || true; GPULOG=; }
trap gpu_log_stop EXIT

make -s all

case "$stage" in
test)
    mkdir -p results/tests
    { env_report; ./build/test; python3 scripts/test_quantize.py 2>&1; } > results/tests/test_output.txt 2>&1
    tail -3 results/tests/test_output.txt
    ;;
tune)
    require_quiet
    mkdir -p results/tuning
    env_report > results/tuning/env.txt
    gpu_log_start results/tuning/gpu_log.csv
    ./build/bench --mode tune --out results/tuning/tune.csv > results/tuning/tune.log 2>&1
    gpu_log_stop
    tail -1 results/tuning/tune.log
    ;;
final)
    require_quiet
    mkdir -p results/final
    env_report > results/final/env.txt
    for i in 1 2 3; do
        ./build/bandwidth > results/final/bandwidth_run$i.txt
        gpu_log_start results/final/gpu_log_run$i.csv
        ./build/bench --mode final --seed "$i" --out results/final/run$i.csv \
            --samples results/final/run${i}_samples.csv > results/final/run$i.log 2>&1
        gpu_log_stop
        gzip -f results/final/run${i}_samples.csv
        tail -1 results/final/run$i.log
    done
    env_report >> results/final/env.txt
    ;;
rotation)
    require_quiet
    mkdir -p results/rotation
    env_report > results/rotation/env_final.txt
    gpu_log_start results/rotation/gpu_log_final.csv
    ./build/bench --mode rotation --out results/rotation/sweep_final.csv \
        --samples results/rotation/sweep_final_samples.csv > results/rotation/sweep_final.log 2>&1
    gpu_log_stop
    gzip -f results/rotation/sweep_final_samples.csv
    tail -1 results/rotation/sweep_final.log
    ;;
nsys)
    require_quiet
    mkdir -p results/nsys
    env_report > results/nsys/env.txt
    nsys --version >> results/nsys/env.txt
    # One final-set pass under CUDA API + kernel tracing (no GPU counters needed). The
    # trace's per-kernel durations cross-check the event timings and name cuBLAS's kernels.
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
        -o results/nsys/final_pass ./build/bench --mode final --seed 9 \
        --out results/nsys/bench_under_nsys.csv > results/nsys/bench_under_nsys.log 2>&1
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
        -o results/nsys/kern_sum results/nsys/final_pass.nsys-rep > /dev/null
    nsys export --force-overwrite=true --type sqlite -o results/nsys/final_pass.sqlite \
        results/nsys/final_pass.nsys-rep > /dev/null
    python3 scripts/nsys_crosscheck.py results/nsys/final_pass.sqlite results/nsys/bench_under_nsys.csv \
        > results/nsys/crosscheck.md
    rm -f results/nsys/final_pass.sqlite
    tail -5 results/nsys/crosscheck.md
    ;;
sweep)
    mkdir -p results/accuracy
    { env_report; python3 scripts/accuracy_sweep.py --out results/accuracy/all_layers.csv; } \
        > results/accuracy/sweep.log 2>&1
    tail -3 results/accuracy/sweep.log
    ;;
*)
    echo "unknown stage $stage" >&2
    exit 2
    ;;
esac
