// Error checking for every CUDA runtime and cuBLAS call.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n", __FILE__,        \
                    __LINE__, #call, cudaGetErrorString(err__));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                           \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %s -> %s\n", __FILE__,      \
                    __LINE__, #call, cublasGetStatusString(st__));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// A kernel launch returns no status; check the launch itself, right after it.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
