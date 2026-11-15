# Nsight Systems cross-check of the isolated event timings

Per bench row: the bench's isolated median (events, under nsys) and the median
per-launch GPU kernel time from the trace (sum of the kernels between spin kernels).

| row | bench iso median (us) | nsys kernel median (us) | bench / nsys | kernels per launch | kernel names |
| --- | --- | --- | --- | --- | --- |
| gate int4 v4 t128_r1_u2_xg_full x67 | 56.29 | 52.26 | 1.077 | 1 | `gemv_vec` |
| gate int4 v4load t128_r1_u2_xg_full x67 | 56.43 | 52.53 | 1.074 | 1 | `gemv_vec` |
| gate int4 read t256_g768 x67 | 55.26 | 51.36 | 1.076 | 1 | `read_kernel` |
| gate bf16 v2abl t128_r1_u1_xs_full x18 | 205.74 | 201.22 | 1.023 | 1 | `gemv_vec` |
| gate bf16 read t256_g768 x18 | 201.62 | 197.55 | 1.021 | 1 | `read_kernel` |
| gate bf16 v2abl t128_r1_u2_xg_full x18 | 204.86 | 201.12 | 1.019 | 1 | `gemv_vec` |
| gate bf16 cublaslt heur1 x18 | 205.02 | 199.94 | 1.025 | 1 | `kernel` |
| gate bf16 v2abl t128_r1_u1_xg_full_ld8 x18 | 204.96 | 201.41 | 1.018 | 1 | `gemv_vec` |
| gate bf16 v2abl t128_r1_u1_xg_pers x18 | 204.82 | 200.78 | 1.020 | 1 | `gemv_vec` |
| gate bf16 v2 t128_r1_u1_xg_full x18 | 206.86 | 202.48 | 1.022 | 1 | `gemv_vec` |
| gate bf16 cublaslt heur2 x18 | 210.94 | 204.86 | 1.030 | 1 | `Kernel2` |
| gate bf16 v1 t256 x18 | 210.06 | 206.56 | 1.017 | 1 | `gemv_v1` |
| gate bf16 cublaslt heur0 x18 | 283.44 | 277.76 | 1.020 | 1 | `kernel` |
| gate bf16 cublas GemmEx_OP_T_32F_DEFAULT x18 | 283.68 | 277.55 | 1.022 | 1 | `kernel` |
| gate bf16 v2abl t256_r1_u1_xg_full x18 | 205.52 | 201.54 | 1.020 | 1 | `gemv_vec` |
| gate bf16 cublaslt heur6 x18 | 289.79 | 282.50 | 1.026 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| gate bf16 v0 t256 x18 | 452.10 | 448.70 | 1.008 | 1 | `gemv_v0` |
| gate bf16 cublaslt heur7 x18 | 401.31 | 395.39 | 1.015 | 1 | `kernel` |
| gate bf16 v2abl t128_r1_u1_xg_full_ld4 x18 | 208.86 | 204.58 | 1.021 | 1 | `gemv_vec` |
| gate bf16 cublaslt heur5 x18 | 218.08 | 209.55 | 1.041 | 2 | `Kernel2`, `splitKreduce_kernel` |
| gate bf16 v2abl t128_r2_u1_xg_full x18 | 205.92 | 201.95 | 1.020 | 1 | `gemv_vec` |
| gate bf16 cublaslt heur4 x18 | 215.97 | 207.58 | 1.040 | 2 | `Kernel2`, `splitKreduce_kernel` |
| gate bf16 v2abl t128_r4_u1_xg_full x18 | 206.03 | 202.35 | 1.018 | 1 | `gemv_vec` |
| gate bf16 v2abl t128_r1_u4_xg_full x18 | 206.94 | 203.07 | 1.019 | 1 | `gemv_vec` |
| gate bf16 cublaslt heur3 x18 | 327.78 | 322.66 | 1.016 | 1 | `kernel` |
| gate int8 read t256_g768 x35 | 103.38 | 99.30 | 1.041 | 1 | `read_kernel` |
| gate int8 v3 t128_r1_u4_xg_full x35 | 104.30 | 100.22 | 1.041 | 1 | `gemv_vec` |
| up int4 v4 t128_r1_u2_xg_full x67 | 57.34 | 53.18 | 1.078 | 1 | `gemv_vec` |
| up int4 v4load t128_r1_u2_xg_full x67 | 56.48 | 52.66 | 1.073 | 1 | `gemv_vec` |
| up int4 read t256_g768 x67 | 55.49 | 51.92 | 1.069 | 1 | `read_kernel` |
| up bf16 v2abl t128_r1_u1_xs_full x18 | 206.72 | 202.53 | 1.021 | 1 | `gemv_vec` |
| up bf16 read t256_g768 x18 | 202.56 | 198.53 | 1.020 | 1 | `read_kernel` |
| up bf16 v2abl t128_r1_u2_xg_full x18 | 205.81 | 201.92 | 1.019 | 1 | `gemv_vec` |
| up bf16 cublaslt heur1 x18 | 206.42 | 200.98 | 1.027 | 1 | `kernel` |
| up bf16 v2abl t128_r1_u1_xg_full_ld8 x18 | 206.00 | 202.27 | 1.018 | 1 | `gemv_vec` |
| up bf16 v2abl t128_r1_u1_xg_pers x18 | 206.77 | 202.06 | 1.023 | 1 | `gemv_vec` |
| up bf16 v2 t128_r1_u1_xg_full x18 | 204.85 | 201.02 | 1.019 | 1 | `gemv_vec` |
| up bf16 cublaslt heur2 x18 | 210.82 | 204.37 | 1.032 | 1 | `Kernel2` |
| up bf16 v1 t256 x18 | 219.86 | 212.86 | 1.033 | 1 | `gemv_v1` |
| up bf16 cublaslt heur0 x18 | 282.82 | 277.78 | 1.018 | 1 | `kernel` |
| up bf16 cublas GemmEx_OP_T_32F_DEFAULT x18 | 283.71 | 277.55 | 1.022 | 1 | `kernel` |
| up bf16 v2abl t256_r1_u1_xg_full x18 | 206.85 | 202.48 | 1.022 | 1 | `gemv_vec` |
| up bf16 cublaslt heur6 x18 | 289.86 | 282.46 | 1.026 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| up bf16 v0 t256 x18 | 451.79 | 448.03 | 1.008 | 1 | `gemv_v0` |
| up bf16 cublaslt heur7 x18 | 400.42 | 395.42 | 1.013 | 1 | `kernel` |
| up bf16 v2abl t128_r1_u1_xg_full_ld4 x18 | 213.95 | 208.48 | 1.026 | 1 | `gemv_vec` |
| up bf16 cublaslt heur5 x18 | 218.10 | 209.94 | 1.039 | 2 | `Kernel2`, `splitKreduce_kernel` |
| up bf16 v2abl t128_r2_u1_xg_full x18 | 205.81 | 201.52 | 1.021 | 1 | `gemv_vec` |
| up bf16 cublaslt heur4 x18 | 216.16 | 208.18 | 1.038 | 2 | `Kernel2`, `splitKreduce_kernel` |
| up bf16 v2abl t128_r4_u1_xg_full x18 | 205.92 | 202.32 | 1.018 | 1 | `gemv_vec` |
| up bf16 v2abl t128_r1_u4_xg_full x18 | 208.82 | 203.47 | 1.026 | 1 | `gemv_vec` |
| up bf16 cublaslt heur3 x18 | 328.72 | 322.91 | 1.018 | 1 | `kernel` |
| up int8 read t256_g768 x35 | 103.39 | 99.02 | 1.044 | 1 | `read_kernel` |
| up int8 v3 t128_r1_u4_xg_full x35 | 103.52 | 99.74 | 1.038 | 1 | `gemv_vec` |
| down int4 v4 t128_r1_u2_xg_full x67 | 65.18 | 60.85 | 1.071 | 1 | `gemv_vec` |
| down int4 v4load t128_r1_u2_xg_full x67 | 58.51 | 54.85 | 1.067 | 1 | `gemv_vec` |
| down int4 read t256_g768 x67 | 55.46 | 51.98 | 1.067 | 1 | `read_kernel` |
| down bf16 v2abl t128_r1_u1_xs_full x18 | 207.82 | 203.66 | 1.020 | 1 | `gemv_vec` |
| down bf16 read t256_g768 x18 | 202.91 | 199.15 | 1.019 | 1 | `read_kernel` |
| down bf16 v2abl t128_r1_u2_xg_full x18 | 208.03 | 204.18 | 1.019 | 1 | `gemv_vec` |
| down bf16 cublaslt heur1 x18 | 295.06 | 289.66 | 1.019 | 1 | `kernel` |
| down bf16 v2abl t128_r1_u1_xg_full_ld8 x18 | 206.99 | 203.12 | 1.019 | 1 | `gemv_vec` |
| down bf16 v2abl t128_r1_u1_xg_pers x18 | 207.79 | 203.52 | 1.021 | 1 | `gemv_vec` |
| down bf16 v2 t128_r1_u1_xg_full x18 | 206.98 | 203.09 | 1.019 | 1 | `gemv_vec` |
| down bf16 cublaslt heur2 x18 | 216.06 | 208.08 | 1.038 | 2 | `Kernel2`, `splitKreduce_kernel` |
| down bf16 v1 t256 x18 | 215.07 | 211.09 | 1.019 | 1 | `gemv_v1` |
| down bf16 cublaslt heur0 x18 | 209.70 | 203.94 | 1.028 | 1 | `kernel` |
| down bf16 cublas GemmEx_OP_T_32F_DEFAULT x18 | 210.08 | 204.13 | 1.029 | 1 | `kernel` |
| down bf16 v2abl t256_r1_u1_xg_full x18 | 206.93 | 203.15 | 1.019 | 1 | `gemv_vec` |
| down bf16 cublaslt heur6 x18 | 223.39 | 214.82 | 1.040 | 2 | `Kernel2`, `splitKreduce_kernel` |
| down bf16 v0 t256 x18 | 988.40 | 984.40 | 1.004 | 1 | `gemv_v0` |
| down bf16 cublaslt heur7 x18 | 226.85 | 221.70 | 1.023 | 2 | `kernel`, `splitKreduce_kernel` |
| down bf16 v2abl t128_r1_u1_xg_full_ld4 x18 | 216.05 | 211.02 | 1.024 | 1 | `gemv_vec` |
| down bf16 cublaslt heur5 x18 | 214.72 | 209.25 | 1.026 | 2 | `kernel`, `splitKreduce_kernel` |
| down bf16 v2abl t128_r2_u1_xg_full x18 | 209.22 | 204.14 | 1.025 | 1 | `gemv_vec` |
| down bf16 cublaslt heur4 x18 | 313.39 | 308.29 | 1.017 | 2 | `kernel`, `splitKreduce_kernel` |
| down bf16 v2abl t128_r4_u1_xg_full x18 | 205.95 | 202.16 | 1.019 | 1 | `gemv_vec` |
| down bf16 v2abl t128_r1_u4_xg_full x18 | 210.03 | 206.16 | 1.019 | 1 | `gemv_vec` |
| down bf16 cublaslt heur3 x18 | 225.09 | 217.12 | 1.037 | 1 | `Kernel2` |
| down int8 read t256_g768 x35 | 103.55 | 99.89 | 1.037 | 1 | `read_kernel` |
| down int8 v3 t128_r1_u4_xg_full x35 | 105.57 | 101.54 | 1.040 | 1 | `gemv_vec` |
| qkv int4 v4 t128_r1_u2_xg_full x75 | 51.01 | 47.02 | 1.085 | 1 | `gemv_vec` |
| qkv int4 v4load t128_r1_u2_xg_full x75 | 50.30 | 46.78 | 1.075 | 1 | `gemv_vec` |
| qkv int4 read t256_g768 x75 | 49.28 | 45.86 | 1.075 | 1 | `read_kernel` |
| qkv bf16 v2abl t128_r1_u1_xs_full x20 | 184.35 | 180.14 | 1.023 | 1 | `gemv_vec` |
| qkv bf16 read t256_g768 x20 | 180.30 | 176.53 | 1.021 | 1 | `read_kernel` |
| qkv bf16 v2abl t128_r1_u2_xg_full x20 | 183.41 | 179.74 | 1.020 | 1 | `gemv_vec` |
| qkv bf16 cublaslt heur1 x20 | 184.32 | 178.88 | 1.030 | 1 | `kernel` |
| qkv bf16 v2abl t128_r1_u1_xg_full_ld8 x20 | 183.49 | 179.98 | 1.019 | 1 | `gemv_vec` |
| qkv bf16 v2abl t128_r1_u1_xg_pers x20 | 183.36 | 179.42 | 1.022 | 1 | `gemv_vec` |
| qkv bf16 v2 t128_r1_u1_xg_full x20 | 183.26 | 179.22 | 1.023 | 1 | `gemv_vec` |
| qkv bf16 cublaslt heur2 x20 | 192.64 | 185.01 | 1.041 | 2 | `Kernel2`, `splitKreduce_kernel` |
| qkv bf16 v1 t256 x20 | 188.34 | 184.50 | 1.021 | 1 | `gemv_v1` |
| qkv bf16 cublaslt heur0 x20 | 253.09 | 248.22 | 1.020 | 1 | `kernel` |
| qkv bf16 cublas GemmEx_OP_T_32F_DEFAULT x20 | 254.05 | 248.26 | 1.023 | 1 | `kernel` |
| qkv bf16 v2abl t256_r1_u1_xg_full x20 | 183.18 | 179.06 | 1.023 | 1 | `gemv_vec` |
| qkv bf16 cublaslt heur6 x20 | 259.07 | 251.42 | 1.030 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| qkv bf16 v0 t256 x20 | 424.94 | 421.14 | 1.009 | 1 | `gemv_v0` |
| qkv bf16 cublaslt heur7 x20 | 359.42 | 354.08 | 1.015 | 1 | `kernel` |
| qkv bf16 v2abl t128_r1_u1_xg_full_ld4 x20 | 185.47 | 181.79 | 1.020 | 1 | `gemv_vec` |
| qkv bf16 cublaslt heur5 x20 | 292.86 | 287.31 | 1.019 | 1 | `kernel` |
| qkv bf16 v2abl t128_r2_u1_xg_full x20 | 183.42 | 179.57 | 1.021 | 1 | `gemv_vec` |
| qkv bf16 cublaslt heur4 x20 | 198.30 | 189.33 | 1.047 | 2 | `Kernel2`, `splitKreduce_kernel` |
| qkv bf16 v2abl t128_r4_u1_xg_full x20 | 184.96 | 181.18 | 1.021 | 1 | `gemv_vec` |
| qkv bf16 v2abl t128_r1_u4_xg_full x20 | 184.24 | 180.16 | 1.023 | 1 | `gemv_vec` |
| qkv bf16 cublaslt heur3 x20 | 188.48 | 182.58 | 1.032 | 1 | `Kernel2` |
| qkv int8 read t256_g768 x39 | 92.22 | 88.29 | 1.045 | 1 | `read_kernel` |
| qkv int8 v3 t128_r1_u4_xg_full x39 | 95.09 | 90.06 | 1.056 | 1 | `gemv_vec` |
| z int4 v4 t128_r1_u2_xg_full x149 | 28.72 | 24.67 | 1.164 | 1 | `gemv_vec` |
| z int4 v4load t128_r1_u2_xg_full x149 | 27.95 | 24.32 | 1.149 | 1 | `gemv_vec` |
| z int4 read t256_g768 x149 | 26.72 | 23.22 | 1.151 | 1 | `read_kernel` |
| z bf16 v2abl t128_r1_u1_xs_full x39 | 95.14 | 90.99 | 1.046 | 1 | `gemv_vec` |
| z bf16 read t256_g768 x39 | 97.55 | 91.44 | 1.067 | 1 | `read_kernel` |
| z bf16 v2abl t128_r1_u2_xg_full x39 | 93.92 | 89.92 | 1.044 | 1 | `gemv_vec` |
| z bf16 cublaslt heur1 x39 | 96.91 | 91.34 | 1.061 | 1 | `kernel` |
| z bf16 v2abl t128_r1_u1_xg_full_ld8 x39 | 93.41 | 89.81 | 1.040 | 1 | `gemv_vec` |
| z bf16 v2abl t128_r1_u1_xg_pers x39 | 96.18 | 90.82 | 1.059 | 1 | `gemv_vec` |
| z bf16 v2 t128_r1_u1_xg_full x39 | 94.24 | 90.29 | 1.044 | 1 | `gemv_vec` |
| z bf16 cublaslt heur2 x39 | 101.52 | 93.97 | 1.080 | 2 | `Kernel2`, `splitKreduce_kernel` |
| z bf16 v1 t256 x39 | 97.22 | 92.91 | 1.046 | 1 | `gemv_v1` |
| z bf16 cublaslt heur0 x39 | 137.68 | 130.54 | 1.055 | 1 | `kernel` |
| z bf16 cublas GemmEx_OP_T_32F_DEFAULT x39 | 132.51 | 126.43 | 1.048 | 1 | `kernel` |
| z bf16 v2abl t256_r1_u1_xg_full x39 | 93.42 | 89.74 | 1.041 | 1 | `gemv_vec` |
| z bf16 cublaslt heur6 x39 | 134.11 | 126.94 | 1.056 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| z bf16 v0 t256 x39 | 278.53 | 274.13 | 1.016 | 1 | `gemv_v0` |
| z bf16 cublaslt heur7 x39 | 187.46 | 182.59 | 1.027 | 1 | `kernel` |
| z bf16 v2abl t128_r1_u1_xg_full_ld4 x39 | 95.39 | 91.57 | 1.042 | 1 | `gemv_vec` |
| z bf16 cublaslt heur5 x39 | 150.46 | 145.10 | 1.037 | 1 | `kernel` |
| z bf16 v2abl t128_r2_u1_xg_full x39 | 94.37 | 90.64 | 1.041 | 1 | `gemv_vec` |
| z bf16 cublaslt heur4 x39 | 101.44 | 95.41 | 1.063 | 1 | `Kernel2` |
| z bf16 v2abl t128_r4_u1_xg_full x39 | 98.48 | 94.29 | 1.044 | 1 | `gemv_vec` |
| z bf16 v2abl t128_r1_u4_xg_full x39 | 94.11 | 89.81 | 1.048 | 1 | `gemv_vec` |
| z bf16 cublaslt heur3 x39 | 102.53 | 94.62 | 1.084 | 2 | `Kernel2`, `splitKreduce_kernel` |
| z int8 read t256_g768 x77 | 48.19 | 44.35 | 1.087 | 1 | `read_kernel` |
| z int8 v3 t128_r1_u4_xg_full x77 | 49.20 | 45.23 | 1.088 | 1 | `gemv_vec` |
| out int4 v4 t128_r1_u2_xg_full x149 | 28.74 | 24.83 | 1.157 | 1 | `gemv_vec` |
| out int4 v4load t128_r1_u2_xg_full x149 | 28.86 | 25.15 | 1.148 | 1 | `gemv_vec` |
| out int4 read t256_g768 x149 | 26.72 | 23.26 | 1.149 | 1 | `read_kernel` |
| out bf16 v2abl t128_r1_u1_xs_full x39 | 95.25 | 91.31 | 1.043 | 1 | `gemv_vec` |
| out bf16 read t256_g768 x39 | 92.85 | 88.85 | 1.045 | 1 | `read_kernel` |
| out bf16 v2abl t128_r1_u2_xg_full x39 | 94.35 | 90.51 | 1.042 | 1 | `gemv_vec` |
| out bf16 cublaslt heur1 x39 | 98.46 | 92.67 | 1.062 | 1 | `kernel` |
| out bf16 v2abl t128_r1_u1_xg_full_ld8 x39 | 95.52 | 91.66 | 1.042 | 1 | `gemv_vec` |
| out bf16 v2abl t128_r1_u1_xg_pers x39 | 95.22 | 91.22 | 1.044 | 1 | `gemv_vec` |
| out bf16 v2 t128_r1_u1_xg_full x39 | 94.11 | 90.13 | 1.044 | 1 | `gemv_vec` |
| out bf16 cublaslt heur2 x39 | 100.42 | 94.19 | 1.066 | 1 | `Kernel2` |
| out bf16 v1 t256 x39 | 99.20 | 95.12 | 1.043 | 1 | `gemv_v1` |
| out bf16 cublaslt heur0 x39 | 146.66 | 138.74 | 1.057 | 1 | `kernel` |
| out bf16 cublas GemmEx_OP_T_32F_DEFAULT x39 | 133.94 | 127.58 | 1.050 | 1 | `kernel` |
| out bf16 v2abl t256_r1_u1_xg_full x39 | 95.47 | 90.82 | 1.051 | 1 | `gemv_vec` |
| out bf16 cublaslt heur6 x39 | 121.73 | 115.01 | 1.058 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| out bf16 v0 t256 x39 | 429.95 | 425.54 | 1.010 | 1 | `gemv_v0` |
| out bf16 cublaslt heur7 x39 | 190.53 | 185.12 | 1.029 | 1 | `kernel` |
| out bf16 v2abl t128_r1_u1_xg_full_ld4 x39 | 97.41 | 93.74 | 1.039 | 1 | `gemv_vec` |
| out bf16 cublaslt heur5 x39 | 145.36 | 139.95 | 1.039 | 1 | `kernel` |
| out bf16 v2abl t128_r2_u1_xg_full x39 | 96.16 | 91.14 | 1.055 | 1 | `gemv_vec` |
| out bf16 cublaslt heur4 x39 | 102.29 | 93.34 | 1.096 | 2 | `Kernel2`, `splitKreduce_kernel` |
| out bf16 v2abl t128_r4_u1_xg_full x39 | 97.47 | 92.43 | 1.055 | 1 | `gemv_vec` |
| out bf16 v2abl t128_r1_u4_xg_full x39 | 95.01 | 90.78 | 1.047 | 1 | `gemv_vec` |
| out bf16 cublaslt heur3 x39 | 103.52 | 94.58 | 1.095 | 2 | `Kernel2`, `splitKreduce_kernel` |
| out int8 read t256_g768 x77 | 48.90 | 44.96 | 1.088 | 1 | `read_kernel` |
| out int8 v3 t128_r1_u4_xg_full x77 | 49.47 | 45.58 | 1.085 | 1 | `gemv_vec` |
| q int4 v4 t128_r1_u2_xg_full x75 | 51.23 | 47.42 | 1.080 | 1 | `gemv_vec` |
| q int4 v4load t128_r1_u2_xg_full x75 | 50.82 | 46.86 | 1.084 | 1 | `gemv_vec` |
| q int4 read t256_g768 x75 | 49.28 | 45.86 | 1.075 | 1 | `read_kernel` |
| q bf16 v2abl t128_r1_u1_xs_full x20 | 185.28 | 181.06 | 1.023 | 1 | `gemv_vec` |
| q bf16 read t256_g768 x20 | 181.34 | 177.57 | 1.021 | 1 | `read_kernel` |
| q bf16 v2abl t128_r1_u2_xg_full x20 | 185.25 | 181.01 | 1.023 | 1 | `gemv_vec` |
| q bf16 cublaslt heur1 x20 | 186.03 | 180.42 | 1.031 | 1 | `kernel` |
| q bf16 v2abl t128_r1_u1_xg_full_ld8 x20 | 184.80 | 180.51 | 1.024 | 1 | `gemv_vec` |
| q bf16 v2abl t128_r1_u1_xg_pers x20 | 183.15 | 179.10 | 1.023 | 1 | `gemv_vec` |
| q bf16 v2 t128_r1_u1_xg_full x20 | 182.53 | 178.67 | 1.022 | 1 | `gemv_vec` |
| q bf16 cublaslt heur2 x20 | 192.46 | 184.54 | 1.043 | 2 | `Kernel2`, `splitKreduce_kernel` |
| q bf16 v1 t256 x20 | 190.08 | 186.08 | 1.021 | 1 | `gemv_v1` |
| q bf16 cublaslt heur0 x20 | 253.74 | 248.21 | 1.022 | 1 | `kernel` |
| q bf16 cublas GemmEx_OP_T_32F_DEFAULT x20 | 253.92 | 247.81 | 1.025 | 1 | `kernel` |
| q bf16 v2abl t256_r1_u1_xg_full x20 | 183.04 | 178.48 | 1.026 | 1 | `gemv_vec` |
| q bf16 cublaslt heur6 x20 | 259.04 | 251.38 | 1.030 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| q bf16 v0 t256 x20 | 420.46 | 416.35 | 1.010 | 1 | `gemv_v0` |
| q bf16 cublaslt heur7 x20 | 359.36 | 354.08 | 1.015 | 1 | `kernel` |
| q bf16 v2abl t128_r1_u1_xg_full_ld4 x20 | 185.18 | 181.15 | 1.022 | 1 | `gemv_vec` |
| q bf16 cublaslt heur5 x20 | 291.86 | 286.64 | 1.018 | 1 | `kernel` |
| q bf16 v2abl t128_r2_u1_xg_full x20 | 183.28 | 179.17 | 1.023 | 1 | `gemv_vec` |
| q bf16 cublaslt heur4 x20 | 195.46 | 186.61 | 1.047 | 2 | `Kernel2`, `splitKreduce_kernel` |
| q bf16 v2abl t128_r4_u1_xg_full x20 | 184.32 | 180.58 | 1.021 | 1 | `gemv_vec` |
| q bf16 v2abl t128_r1_u4_xg_full x20 | 183.33 | 179.28 | 1.023 | 1 | `gemv_vec` |
| q bf16 cublaslt heur3 x20 | 188.35 | 182.29 | 1.033 | 1 | `Kernel2` |
| q int8 read t256_g768 x39 | 92.06 | 87.74 | 1.049 | 1 | `read_kernel` |
| q int8 v3 t128_r1_u4_xg_full x39 | 92.19 | 88.30 | 1.044 | 1 | `gemv_vec` |
| k int4 v4 t128_r1_u2_xg_full x596 | 11.36 | 7.87 | 1.443 | 1 | `gemv_vec` |
| k int4 v4load t128_r1_u2_xg_full x596 | 11.36 | 7.52 | 1.511 | 1 | `gemv_vec` |
| k int4 read t256_g768 x596 | 10.37 | 6.98 | 1.486 | 1 | `read_kernel` |
| k bf16 v2abl t128_r1_u1_xs_full x154 | 27.46 | 23.47 | 1.170 | 1 | `gemv_vec` |
| k bf16 read t256_g768 x154 | 26.46 | 22.59 | 1.171 | 1 | `read_kernel` |
| k bf16 v2abl t128_r1_u2_xg_full x154 | 26.82 | 23.17 | 1.157 | 1 | `gemv_vec` |
| k bf16 cublaslt heur1 x154 | 29.31 | 23.71 | 1.236 | 1 | `kernel` |
| k bf16 v2abl t128_r1_u1_xg_full_ld8 x154 | 26.72 | 23.26 | 1.149 | 1 | `gemv_vec` |
| k bf16 v2abl t128_r1_u1_xg_pers x154 | 26.72 | 23.10 | 1.157 | 1 | `gemv_vec` |
| k bf16 v2 t128_r1_u1_xg_full x154 | 28.51 | 24.53 | 1.162 | 1 | `gemv_vec` |
| k bf16 cublaslt heur2 x154 | 40.38 | 35.01 | 1.154 | 1 | `kernel` |
| k bf16 v1 t256 x154 | 27.65 | 23.82 | 1.161 | 1 | `gemv_v1` |
| k bf16 cublaslt heur0 x154 | 39.82 | 34.14 | 1.166 | 1 | `kernel` |
| k bf16 cublas GemmEx_OP_T_32F_DEFAULT x154 | 40.86 | 34.05 | 1.200 | 1 | `kernel` |
| k bf16 v2abl t256_r1_u1_xg_full x154 | 27.23 | 23.10 | 1.179 | 1 | `gemv_vec` |
| k bf16 cublaslt heur6 x154 | 39.97 | 33.25 | 1.202 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| k bf16 v0 t256 x154 | 275.55 | 271.81 | 1.014 | 1 | `gemv_v0` |
| k bf16 cublaslt heur7 x154 | 58.18 | 52.24 | 1.114 | 1 | `kernel` |
| k bf16 v2abl t128_r1_u1_xg_full_ld4 x154 | 27.74 | 24.00 | 1.156 | 1 | `gemv_vec` |
| k bf16 cublaslt heur5 x154 | 35.74 | 27.10 | 1.319 | 2 | `Kernel2`, `splitKreduce_kernel` |
| k bf16 v2abl t128_r2_u1_xg_full x154 | 27.58 | 23.42 | 1.178 | 1 | `gemv_vec` |
| k bf16 cublaslt heur4 x154 | 33.98 | 25.66 | 1.324 | 2 | `Kernel2`, `splitKreduce_kernel` |
| k bf16 v2abl t128_r4_u1_xg_full x154 | 27.68 | 23.74 | 1.166 | 1 | `gemv_vec` |
| k bf16 v2abl t128_r1_u4_xg_full x154 | 26.72 | 23.17 | 1.153 | 1 | `gemv_vec` |
| k bf16 cublaslt heur3 x154 | 36.86 | 30.48 | 1.209 | 1 | `Kernel2` |
| k int8 read t256_g768 x307 | 15.36 | 11.74 | 1.308 | 1 | `read_kernel` |
| k int8 v3 t128_r1_u4_xg_full x307 | 19.62 | 16.13 | 1.216 | 1 | `gemv_vec` |
| v int4 v4 t128_r1_u2_xg_full x596 | 11.39 | 7.87 | 1.447 | 1 | `gemv_vec` |
| v int4 v4load t128_r1_u2_xg_full x596 | 11.42 | 7.58 | 1.506 | 1 | `gemv_vec` |
| v int4 read t256_g768 x596 | 11.49 | 8.13 | 1.413 | 1 | `read_kernel` |
| v bf16 v2abl t128_r1_u1_xs_full x154 | 27.52 | 23.46 | 1.173 | 1 | `gemv_vec` |
| v bf16 read t256_g768 x154 | 25.79 | 22.53 | 1.145 | 1 | `read_kernel` |
| v bf16 v2abl t128_r1_u2_xg_full x154 | 28.74 | 25.02 | 1.148 | 1 | `gemv_vec` |
| v bf16 cublaslt heur1 x154 | 28.74 | 23.71 | 1.212 | 1 | `kernel` |
| v bf16 v2abl t128_r1_u1_xg_full_ld8 x154 | 27.65 | 23.71 | 1.166 | 1 | `gemv_vec` |
| v bf16 v2abl t128_r1_u1_xg_pers x154 | 26.75 | 23.12 | 1.157 | 1 | `gemv_vec` |
| v bf16 v2 t128_r1_u1_xg_full x154 | 26.72 | 23.07 | 1.158 | 1 | `gemv_vec` |
| v bf16 cublaslt heur2 x154 | 40.67 | 35.01 | 1.162 | 1 | `kernel` |
| v bf16 v1 t256 x154 | 27.65 | 23.78 | 1.163 | 1 | `gemv_v1` |
| v bf16 cublaslt heur0 x154 | 39.76 | 34.06 | 1.167 | 1 | `kernel` |
| v bf16 cublas GemmEx_OP_T_32F_DEFAULT x154 | 40.74 | 34.18 | 1.192 | 1 | `kernel` |
| v bf16 v2abl t256_r1_u1_xg_full x154 | 26.69 | 23.04 | 1.158 | 1 | `gemv_vec` |
| v bf16 cublaslt heur6 x154 | 40.00 | 33.57 | 1.192 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| v bf16 v0 t256 x154 | 275.42 | 271.39 | 1.015 | 1 | `gemv_v0` |
| v bf16 cublaslt heur7 x154 | 58.43 | 52.99 | 1.103 | 1 | `kernel` |
| v bf16 v2abl t128_r1_u1_xg_full_ld4 x154 | 27.62 | 23.58 | 1.171 | 1 | `gemv_vec` |
| v bf16 cublaslt heur5 x154 | 35.81 | 27.07 | 1.323 | 2 | `Kernel2`, `splitKreduce_kernel` |
| v bf16 v2abl t128_r2_u1_xg_full x154 | 27.58 | 23.42 | 1.178 | 1 | `gemv_vec` |
| v bf16 cublaslt heur4 x154 | 34.50 | 25.84 | 1.335 | 2 | `Kernel2`, `splitKreduce_kernel` |
| v bf16 v2abl t128_r4_u1_xg_full x154 | 27.71 | 23.71 | 1.169 | 1 | `gemv_vec` |
| v bf16 v2abl t128_r1_u4_xg_full x154 | 27.30 | 23.15 | 1.179 | 1 | `gemv_vec` |
| v bf16 cublaslt heur3 x154 | 34.75 | 28.46 | 1.221 | 1 | `Kernel2` |
| v int8 read t256_g768 x307 | 15.42 | 11.74 | 1.313 | 1 | `read_kernel` |
| v int8 v3 t128_r1_u4_xg_full x307 | 15.78 | 12.24 | 1.289 | 1 | `gemv_vec` |
| o int4 v4 t128_r1_u2_xg_full x149 | 28.64 | 24.80 | 1.155 | 1 | `gemv_vec` |
| o int4 v4load t128_r1_u2_xg_full x149 | 28.67 | 24.61 | 1.165 | 1 | `gemv_vec` |
| o int4 read t256_g768 x149 | 28.83 | 25.12 | 1.148 | 1 | `read_kernel` |
| o bf16 v2abl t128_r1_u1_xs_full x39 | 95.14 | 90.85 | 1.047 | 1 | `gemv_vec` |
| o bf16 read t256_g768 x39 | 92.02 | 88.00 | 1.046 | 1 | `read_kernel` |
| o bf16 v2abl t128_r1_u2_xg_full x39 | 93.49 | 89.82 | 1.041 | 1 | `gemv_vec` |
| o bf16 cublaslt heur1 x39 | 96.22 | 90.77 | 1.060 | 1 | `kernel` |
| o bf16 v2abl t128_r1_u1_xg_full_ld8 x39 | 94.37 | 90.26 | 1.046 | 1 | `gemv_vec` |
| o bf16 v2abl t128_r1_u1_xg_pers x39 | 93.92 | 89.89 | 1.045 | 1 | `gemv_vec` |
| o bf16 v2 t128_r1_u1_xg_full x39 | 93.42 | 89.82 | 1.040 | 1 | `gemv_vec` |
| o bf16 cublaslt heur2 x39 | 100.22 | 93.44 | 1.073 | 1 | `Kernel2` |
| o bf16 v1 t256 x39 | 98.16 | 94.02 | 1.044 | 1 | `gemv_v1` |
| o bf16 cublaslt heur0 x39 | 132.22 | 127.09 | 1.040 | 1 | `kernel` |
| o bf16 cublas GemmEx_OP_T_32F_DEFAULT x39 | 134.34 | 128.50 | 1.045 | 1 | `kernel` |
| o bf16 v2abl t256_r1_u1_xg_full x39 | 94.00 | 89.71 | 1.048 | 1 | `gemv_vec` |
| o bf16 cublaslt heur6 x39 | 105.54 | 99.33 | 1.062 | 2 | `dot_kernel`, `reduce_1Block_kernel` |
| o bf16 v0 t256 x39 | 429.94 | 425.57 | 1.010 | 1 | `gemv_v0` |
| o bf16 cublaslt heur7 x39 | 190.34 | 185.06 | 1.029 | 1 | `kernel` |
| o bf16 v2abl t128_r1_u1_xg_full_ld4 x39 | 96.37 | 92.70 | 1.040 | 1 | `gemv_vec` |
| o bf16 cublaslt heur5 x39 | 144.46 | 139.15 | 1.038 | 1 | `kernel` |
| o bf16 v2abl t128_r2_u1_xg_full x39 | 94.24 | 90.14 | 1.045 | 1 | `gemv_vec` |
| o bf16 cublaslt heur4 x39 | 101.44 | 93.01 | 1.091 | 2 | `Kernel2`, `splitKreduce_kernel` |
| o bf16 v2abl t128_r4_u1_xg_full x39 | 94.86 | 90.75 | 1.045 | 1 | `gemv_vec` |
| o bf16 v2abl t128_r1_u4_xg_full x39 | 93.25 | 89.41 | 1.043 | 1 | `gemv_vec` |
| o bf16 cublaslt heur3 x39 | 102.26 | 93.15 | 1.098 | 2 | `Kernel2`, `splitKreduce_kernel` |
| o int8 read t256_g768 x77 | 49.07 | 44.91 | 1.093 | 1 | `read_kernel` |
| o int8 v3 t128_r1_u4_xg_full x77 | 49.12 | 45.18 | 1.087 | 1 | `gemv_vec` |
| lmhead int4 v4 t128_r1_u2_xg_full x3 | 1381.18 | 1376.96 | 1.003 | 1 | `gemv_vec` |
| lmhead int4 v4load t128_r1_u2_xg_full x3 | 1382.18 | 1377.52 | 1.003 | 1 | `gemv_vec` |
| lmhead int4 read t256_g768 x3 | 1385.49 | 1381.01 | 1.003 | 1 | `read_kernel` |
| lmhead bf16 v2abl t128_r1_u1_xs_full x2 | 5347.10 | 5342.67 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 read t256_g768 x2 | 5387.73 | 5383.15 | 1.001 | 1 | `read_kernel` |
| lmhead bf16 v2abl t128_r1_u2_xg_full x2 | 5369.90 | 5365.79 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 cublaslt heur1 x2 | 7359.52 | 7353.38 | 1.001 | 1 | `kernel` |
| lmhead bf16 v2abl t128_r1_u1_xg_full_ld8 x2 | 5373.87 | 5369.38 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 v2abl t128_r1_u1_xg_pers x2 | 5434.35 | 5429.95 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 v2 t128_r1_u1_xg_full x2 | 5354.45 | 5350.05 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 cublaslt heur2 x2 | 5393.23 | 5386.53 | 1.001 | 1 | `Kernel2` |
| lmhead bf16 v1 t256 x2 | 5469.06 | 5464.50 | 1.001 | 1 | `gemv_v1` |
| lmhead bf16 cublaslt heur0 x2 | 7360.43 | 7355.34 | 1.001 | 1 | `kernel` |
| lmhead bf16 cublas GemmEx_OP_T_32F_DEFAULT x2 | 7359.31 | 7352.37 | 1.001 | 1 | `kernel` |
| lmhead bf16 v2abl t256_r1_u1_xg_full x2 | 5345.26 | 5341.34 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 cublaslt heur6 x2 | 9340.66 | 9335.30 | 1.001 | 1 | `kernel` |
| lmhead bf16 v0 t256 x2 | 11187.66 | 11184.18 | 1.000 | 1 | `gemv_v0` |
| lmhead bf16 cublaslt heur7 x2 | 11758.42 | 11752.18 | 1.001 | 1 | `kernel` |
| lmhead bf16 v2abl t128_r1_u1_xg_full_ld4 x2 | 5413.01 | 5408.72 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 cublaslt heur5 x2 | 8702.78 | 8696.35 | 1.001 | 1 | `kernel` |
| lmhead bf16 v2abl t128_r2_u1_xg_full x2 | 5360.72 | 5356.83 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 cublaslt heur4 x2 | 5611.31 | 5604.72 | 1.001 | 1 | `Kernel2` |
| lmhead bf16 v2abl t128_r4_u1_xg_full x2 | 5388.74 | 5384.96 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 v2abl t128_r1_u4_xg_full x2 | 5418.22 | 5414.13 | 1.001 | 1 | `gemv_vec` |
| lmhead bf16 cublaslt heur3 x2 | 5306.35 | 5300.56 | 1.001 | 1 | `kernel` |
| lmhead int8 read t256_g768 x2 | 2693.90 | 2688.24 | 1.002 | 1 | `read_kernel` |
| lmhead int8 v3 t128_r1_u4_xg_full x2 | 2660.85 | 2656.46 | 1.002 | 1 | `gemv_vec` |

297 rows; bench/nsys ratio: min 1.000, median 1.040, max 1.511 (events add the event-record overhead around the kernel).
