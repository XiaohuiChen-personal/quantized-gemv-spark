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
#   scripts/run_stage.sh marlin    vLLM Marlin INT4 at batch 1 (H4)  -> results/marlin/
#   scripts/run_stage.sh sweep     all-layer accuracy sweep (CPU)    -> results/accuracy/
#
# Added after the freeze at c1d91fc (new directories; the frozen results are not touched):
#
#   scripts/run_stage.sh xsample   256 random x for every large matrix of all 32 layers
#                                  (CPU)                             -> results/accuracy/
#   scripts/run_stage.sh ncu       Nsight Compute (needs sudo here)  -> results/ncu/
#   scripts/run_stage.sh realx     real activations: capture (vLLM image, transformers),
#                                  INT8/INT4 error (CPU), GPU kernels on real x
#                                                                    -> results/accuracy/real_x/
#   scripts/run_stage.sh e2e       INT4/INT8 GPTQ checkpoints from this quantizer, then
#                                  vLLM decode tokens/s and perplexity, BF16/INT8/INT4,
#                                  three executions                  -> results/e2e/
#
# Every stage records provenance (UTC time, host, git commit, GPU state) and refuses to
# time anything while another process uses the GPU. Timed stages log GPU clocks, power
# and temperature once a second next to their results.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export OMP_WAIT_POLICY=PASSIVE  # OpenMP threads must not spin during GPU timing
cd "$(dirname "$0")/.."
stage=${1:?usage: run_stage.sh test|tune|final|rotation|nsys|marlin|sweep|xsample|ncu|realx|e2e}

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
# python3 in the vLLM 0.28 image (transformers 5.15, PyTorch 2.13, CUDA 13.0), as this user,
# with the Hugging Face cache at /hf and the repository at /work. (The vllm-node-qwen35
# image of the Marlin stage cannot load the full Qwen3.5 model: its vLLM 0.16 pre-release
# fails parsing the config with its own transformers.) Offline: every input is fetched before
# (scripts/fetch_extras.py). git cannot describe the mounted repository inside the
# container (another uid), so the description is passed in GIT_DESC.
VLLM_IMAGE=vllm/vllm-openai:v0.28.0-aarch64
docker_py() {
    mkdir -p ~/tmp/en605617/vllm-cache
    docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
        -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/vllm-cache -e GIT_DESC="$(git describe --always --dirty)" \
        -v "$HOME/.cache/huggingface:/hf" -v ~/tmp/en605617/vllm-cache:/vllm-cache -v "$PWD":/work -w /work \
        --entrypoint python3 "$VLLM_IMAGE" "$@"
}

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
marlin)
    require_quiet
    mkdir -p results/marlin
    env_report > results/marlin/env.txt
    gpu_log_start results/marlin/gpu_log.csv
    # Three executions with vLLM's defaults (H4), then one with the atomic-add reduction its
    # log recommends for small N (a variant, reported alongside).
    for run in 1 2 3 atomic; do
        extra=""; [ "$run" = atomic ] && extra="-e VLLM_MARLIN_USE_ATOMIC_ADD=1"
        docker run --rm --gpus all --ipc=host -e GIT_DESC="$(git describe --always --dirty)" $extra \
            -v "$PWD":/work -w /work vllm-node-qwen35:latest \
            python3 scripts/marlin_bench.py --out results/marlin/marlin_run$run.csv > results/marlin/marlin_run$run.log 2>&1
        grep -E '^(gate|down|lmhead) ' results/marlin/marlin_run$run.log
    done
    gpu_log_stop
    ;;
sweep)
    mkdir -p results/accuracy
    { env_report; python3 scripts/accuracy_sweep.py --out results/accuracy/all_layers.csv; } \
        > results/accuracy/sweep.log 2>&1
    tail -3 results/accuracy/sweep.log
    ;;
xsample)
    mkdir -p results/accuracy
    { env_report; python3 scripts/x_sample_check.py --all-layers --out results/accuracy/x_samples_all_layers.csv; } \
        > results/accuracy/x_samples_all_layers.log 2>&1
    tail -1 results/accuracy/x_samples_all_layers.log
    ;;
ncu)
    require_quiet
    mkdir -p results/ncu ~/tmp/en605617/ncu
    env_report > results/ncu/env.txt
    ncu --version | tail -1 >> results/ncu/env.txt
    # GPU performance counters are admin-only on this machine (RmProfilingAdminOnly: 1),
    # so the profiler runs under sudo. bench --mode ncu makes five isolated launches per
    # (matrix, kernel); only those are profiled (NVTX filter). Nsight Compute flushes the
    # caches before every replay pass, so each profiled launch reads its operand from
    # DRAM; clocks are left as they are (--clock-control none). The GB10 has no dram__*
    # counters: DRAM reads are the L2 sectors filled from system memory.
    rep=~/tmp/en605617/ncu/ncu_final
    sudo -n env OMP_WAIT_POLICY=PASSIVE "$(command -v ncu)" --nvtx --nvtx-include "regex:^iso /" \
        --clock-control none --cache-control all \
        --section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats \
        --section MemoryWorkloadAnalysis \
        --metrics lts__d_sectors_fill_sysmem.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_bytes.sum,gpu__time_duration.sum,smsp__inst_executed.sum \
        -f -o "$rep" ./build/bench --mode ncu --only gate,down,qkv,z,out,k,lmhead \
        --out results/ncu/bench_under_ncu.csv > results/ncu/bench_under_ncu.log 2>&1
    sudo -n chown -R "$(id -u):$(id -g)" results/ncu ~/tmp/en605617/ncu
    ncu --import "$rep.ncu-rep" --csv --page raw --print-units base > results/ncu/ncu_raw.csv
    python3 scripts/ncu_summary.py results/ncu/ncu_raw.csv results/ncu/bench_under_ncu.csv \
        --csv results/ncu/summary.csv > results/ncu/summary.md
    gzip -f results/ncu/ncu_raw.csv
    tail -1 results/ncu/bench_under_ncu.log
    head -3 results/ncu/summary.md | tail -1
    ;;
realx)
    require_quiet
    out=results/accuracy/real_x
    mkdir -p "$out/gpu"
    [ -f data/wikitext2_test.txt ] || python3 scripts/fetch_extras.py
    env_report > "$out/env.txt"
    docker_py scripts/capture_activations.py --out data/real_x > "$out/capture.log" 2>&1
    cp data/real_x/manifest.json "$out/capture_manifest.json"
    rm -rf data/real_x_gpu
    python3 scripts/real_x_accuracy.py --gpu-dirs data/real_x_gpu --out "$out" > "$out/accuracy.log" 2>&1
    # The GPU kernels on two real x per benchmarked matrix: correctness checks and error.
    : > "$out/gpu.log"
    for d in data/real_x_gpu/*/; do
        id=$(basename "$d")
        ./build/bench --mode quick --data "$d" --out "$out/gpu/$id.csv" >> "$out/gpu.log" 2>&1
    done
    python3 scripts/real_x_accuracy.py --collect-gpu "$out" | tee -a "$out/gpu.log"
    tail -1 "$out/capture.log"
    tail -1 "$out/accuracy.log"
    ;;
e2e)
    require_quiet
    out=results/e2e
    mkdir -p "$out" data/ckpt ~/tmp/en605617/e2e
    [ -f data/wikitext2_test.txt ] || python3 scripts/fetch_extras.py
    env_report > "$out/env.txt"
    for b in 4 8; do
        if [ ! -f data/ckpt/int$b/quantization_info.json ]; then
            python3 scripts/make_quant_checkpoint.py --bits $b --out data/ckpt/int$b --verify-data data \
                > "$out/ckpt_int$b.log" 2>&1
        fi
        cp data/ckpt/int$b/quantization_info.json "$out/ckpt_int$b.json"
    done
    bf16=/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    gpu_log_start "$out/gpu_log.csv"
    for run in 1 2 3; do  # three executions, the order of the formats shuffled per execution
        order=$(python3 -c "import random; f = ['bf16', 'int8', 'int4']; random.Random($run).shuffle(f); print(*f)")
        for label in $order; do
            case $label in
                bf16) model=$bf16 ;;
                *) model=/work/data/ckpt/$label ;;
            esac
            if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
                echo "another process is using the GPU before $label run $run" >&2; exit 1
            fi
            docker_py scripts/e2e_decode.py --model "$model" --label "$label" --reps 5 --warmup 2 \
                --out "$out/${label}_run$run.json" > "$out/${label}_run$run.log" 2>&1
            grep -E "decode tokens/s|perplexity" "$out/${label}_run$run.log"
        done
    done
    gpu_log_stop
    # GPU kernel breakdown of the decode steps: one request of 1 + 256 tokens per format
    # under Nsight Systems (the host's installation, mounted into the container; CUDA-graph
    # kernels traced one by one), vLLM's engine in-process so the profiler API covers it.
    NSYS_DIR=/opt/nvidia/nsight-systems
    nsys_bin=$(readlink -f "$(command -v nsys)")
    for label in bf16 int8 int4; do
        case $label in
            bf16) model=$bf16 ;;
            *) model=/work/data/ckpt/$label ;;
        esac
        docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
            -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/vllm-cache -e GIT_DESC="$(git describe --always --dirty)" \
            -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -v "$HOME/.cache/huggingface:/hf" \
            -v ~/tmp/en605617/vllm-cache:/vllm-cache -v "$NSYS_DIR:$NSYS_DIR:ro" -v ~/tmp/en605617/e2e:/prof \
            -v "$PWD":/work -w /work --entrypoint "$nsys_bin" "$VLLM_IMAGE" \
            profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
            --capture-range-end=stop --sample=none --cpuctxsw=none --force-overwrite=true -o "/prof/decode_$label" \
            python3 scripts/e2e_decode.py --model "$model" --label "$label" --contexts 128 --warmup 2 \
            --profile-decode 256 --out "$out/profile_$label.json" > "$out/profile_$label.log" 2>&1
        nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
            -o "$out/profile_$label" ~/tmp/en605617/e2e/decode_$label.nsys-rep > /dev/null
        grep -E "profiled" "$out/profile_$label.log"
    done
    env_report >> "$out/env.txt"
    ;;
*)
    echo "unknown stage $stage" >&2
    exit 2
    ;;
esac
