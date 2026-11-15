# Kernels cuBLAS / cuBLASLt ran (Nsight Systems, isolated range of each bench row)

| row | grid | block | demangled kernel name (truncated) |
| --- | --- | --- | --- |
| down bf16 cublas GemmEx_OP_T_32F_DEFAULT x18 | 640x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| down bf16 cublaslt heur0 x18 | 640x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| down bf16 cublaslt heur1 x18 | 640x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| down bf16 cublaslt heur2 x18 | 8x20x11 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x1_tn_align8>(T1::Params)` |
| down bf16 cublaslt heur3 x18 | 8x20x14 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_64x2_tn_align8>(T1::Params)` |
| down bf16 cublaslt heur4 x18 | 320x1x7 | 16x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)7, (boo` |
| down bf16 cublaslt heur5 x18 | 640x1x4 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)7, (boo` |
| down bf16 cublaslt heur6 x18 | 8x20x3 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_64x2_tn_align8>(T1::Params)` |
| down bf16 cublaslt heur7 x18 | 320x1x4 | 32x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)8, (boo` |
| gate bf16 cublas GemmEx_OP_T_32F_DEFAULT x18 | 2304x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| gate bf16 cublaslt heur0 x18 | 2304x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| gate bf16 cublaslt heur1 x18 | 2304x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| gate bf16 cublaslt heur2 x18 | 8x36x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| gate bf16 cublaslt heur3 x18 | 1152x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| gate bf16 cublaslt heur4 x18 | 8x72x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| gate bf16 cublaslt heur5 x18 | 8x72x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| gate bf16 cublaslt heur6 x18 | 14x1x9216 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| gate bf16 cublaslt heur7 x18 | 2304x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| k bf16 cublas GemmEx_OP_T_32F_DEFAULT x154 | 256x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| k bf16 cublaslt heur0 x154 | 256x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| k bf16 cublaslt heur1 x154 | 256x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| k bf16 cublaslt heur2 x154 | 128x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| k bf16 cublaslt heur3 x154 | 8x4x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| k bf16 cublaslt heur4 x154 | 8x8x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| k bf16 cublaslt heur5 x154 | 8x8x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| k bf16 cublaslt heur6 x154 | 14x1x1024 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| k bf16 cublaslt heur7 x154 | 256x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublas GemmEx_OP_T_32F_DEFAULT x2 | 31040x1x1 | 16x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublaslt heur0 x2 | 31040x1x1 | 16x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublaslt heur1 x2 | 62080x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublaslt heur2 x2 | 8x970x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(T1::Params)` |
| lmhead bf16 cublaslt heur3 x2 | 62080x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublaslt heur4 x2 | 8x970x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8>(T1::Params)` |
| lmhead bf16 cublaslt heur5 x2 | 31040x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublaslt heur6 x2 | 62080x1x1 | 8x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| lmhead bf16 cublaslt heur7 x2 | 62080x1x1 | 32x4x1 | `std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)1, (bool)1, (bool)1, (bool)` |
| o bf16 cublas GemmEx_OP_T_32F_DEFAULT x39 | 640x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| o bf16 cublaslt heur0 x39 | 640x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| o bf16 cublaslt heur1 x39 | 640x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| o bf16 cublaslt heur2 x39 | 8x10x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| o bf16 cublaslt heur3 x39 | 8x20x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| o bf16 cublaslt heur4 x39 | 8x20x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| o bf16 cublaslt heur5 x39 | 320x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| o bf16 cublaslt heur6 x39 | 14x1x2560 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| o bf16 cublaslt heur7 x39 | 640x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| out bf16 cublas GemmEx_OP_T_32F_DEFAULT x39 | 640x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| out bf16 cublaslt heur0 x39 | 640x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| out bf16 cublaslt heur1 x39 | 640x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| out bf16 cublaslt heur2 x39 | 8x10x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| out bf16 cublaslt heur3 x39 | 8x20x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| out bf16 cublaslt heur4 x39 | 8x20x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| out bf16 cublaslt heur5 x39 | 320x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| out bf16 cublaslt heur6 x39 | 14x1x2560 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| out bf16 cublaslt heur7 x39 | 640x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| q bf16 cublas GemmEx_OP_T_32F_DEFAULT x20 | 2048x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| q bf16 cublaslt heur0 x20 | 2048x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| q bf16 cublaslt heur1 x20 | 2048x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| q bf16 cublaslt heur2 x20 | 8x64x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| q bf16 cublaslt heur3 x20 | 8x32x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| q bf16 cublaslt heur4 x20 | 8x64x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| q bf16 cublaslt heur5 x20 | 1024x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| q bf16 cublaslt heur6 x20 | 14x1x8192 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| q bf16 cublaslt heur7 x20 | 2048x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| qkv bf16 cublas GemmEx_OP_T_32F_DEFAULT x20 | 2048x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| qkv bf16 cublaslt heur0 x20 | 2048x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| qkv bf16 cublaslt heur1 x20 | 2048x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| qkv bf16 cublaslt heur2 x20 | 8x64x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| qkv bf16 cublaslt heur3 x20 | 8x32x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| qkv bf16 cublaslt heur4 x20 | 8x64x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| qkv bf16 cublaslt heur5 x20 | 1024x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| qkv bf16 cublaslt heur6 x20 | 14x1x8192 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| qkv bf16 cublaslt heur7 x20 | 2048x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| up bf16 cublas GemmEx_OP_T_32F_DEFAULT x18 | 2304x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| up bf16 cublaslt heur0 x18 | 2304x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| up bf16 cublaslt heur1 x18 | 2304x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| up bf16 cublaslt heur2 x18 | 8x36x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| up bf16 cublaslt heur3 x18 | 1152x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| up bf16 cublaslt heur4 x18 | 8x72x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| up bf16 cublaslt heur5 x18 | 8x72x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| up bf16 cublaslt heur6 x18 | 14x1x9216 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| up bf16 cublaslt heur7 x18 | 2304x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| v bf16 cublas GemmEx_OP_T_32F_DEFAULT x154 | 256x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| v bf16 cublaslt heur0 x154 | 256x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| v bf16 cublaslt heur1 x154 | 256x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| v bf16 cublaslt heur2 x154 | 128x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| v bf16 cublaslt heur3 x154 | 8x4x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| v bf16 cublaslt heur4 x154 | 8x8x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| v bf16 cublaslt heur5 x154 | 8x8x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| v bf16 cublaslt heur6 x154 | 14x1x1024 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| v bf16 cublaslt heur7 x154 | 256x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| z bf16 cublas GemmEx_OP_T_32F_DEFAULT x39 | 1024x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| z bf16 cublaslt heur0 x39 | 1024x1x1 | 16x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| z bf16 cublaslt heur1 x39 | 1024x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| z bf16 cublaslt heur2 x39 | 8x32x7 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| z bf16 cublaslt heur3 x39 | 8x32x4 | 32x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` |
| z bf16 cublaslt heur4 x39 | 8x16x1 | 128x1x1 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(T1::Params)` |
| z bf16 cublaslt heur5 x39 | 512x1x1 | 8x8x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
| z bf16 cublaslt heur6 x39 | 14x1x4096 | 128x1x1 | `void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<float>>>(` |
| z bf16 cublaslt heur7 x39 | 1024x1x1 | 32x4x1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool` |
