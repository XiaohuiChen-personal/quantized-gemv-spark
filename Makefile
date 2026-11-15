# Trading Bits for Bandwidth: quantized GEMV kernels on the DGX Spark (GB10, sm_121).
#
#   make            build bench, test and bandwidth into build/
#   make test       unit tests: CUDA kernels (build/test) and the Python quantizer
#   make data       export the Qwen3.5-4B matrices, quantize them, write the x vectors
#   make bench      one final-mode benchmark run into results/dev/ (development)
#   make clean
#
# The full measurement pipeline (tuning, final sweeps, analysis) is scripts/run_stage.sh.
# ARCH defaults to the GPU in this machine; on a machine without one, pass e.g.
# ARCH=-arch=sm_121.

NVCC     ?= nvcc
ARCH     ?= -arch=native
PYTHON   ?= python3
GIT_DESC := $(shell git describe --always --dirty 2>/dev/null || echo unknown)
NVFLAGS  := -O3 -std=c++20 -lineinfo $(ARCH) -DGIT_DESC='"$(GIT_DESC)"' \
            -Xcompiler -fopenmp,-Wall,-Wextra
LIBS     := -lcublas -lcublasLt -lgomp
HEADERS  := $(wildcard src/*.h src/*.cuh)

# Quiet OpenMP threads after the CPU reference: they must not spin during GPU timing,
# because the CPU and the GPU share the LPDDR5x bus.
export OMP_WAIT_POLICY := PASSIVE

.PHONY: all test data bench clean

all: build/bench build/test build/bandwidth

build:
	mkdir -p build

build/bench: src/bench.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) -o $@ src/bench.cu $(LIBS)

build/test: src/test.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) -o $@ src/test.cu $(LIBS)

build/bandwidth: src/bandwidth.cu src/cuda_check.h | build
	$(NVCC) $(NVFLAGS) -o $@ src/bandwidth.cu

test: build/test
	./build/test
	$(PYTHON) scripts/test_quantize.py

data:
	$(PYTHON) scripts/export_weights.py
	$(PYTHON) scripts/quantize.py
	$(PYTHON) scripts/make_x.py

bench: build/bench
	mkdir -p results/dev
	./build/bench --mode final --out results/dev/bench.csv

clean:
	rm -rf build
