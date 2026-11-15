#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
__global__ void read_k(const float4* p, size_t n, float* sink){ float s=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i]; s+=v.x+v.y+v.z+v.w;} if(s==1234.5f)*sink=s; }
__global__ void spin_gt(unsigned long long ns){ unsigned long long t0,t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t0)); do{asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));}while(t-t0<ns); }
int main(){
  float4* d; float* sink; size_t maxb = 512ull<<20; cudaMalloc(&d, maxb); cudaMemset(d, 1, maxb); cudaMalloc(&sink, 4);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (size_t mb : {4, 20, 100}) {   // L2-resident, ~L2, DRAM
    size_t n = mb*(1<<20)/16;
    for (int mode=0; mode<2; ++mode) {
      char lab[64]; snprintf(lab, 64, "%s %zuMB", mode?"spin":"cold", mb); nvtxRangePushA(lab);
      std::vector<double> t;
      for (int i=0;i<200;++i){ if (mode) spin_gt<<<1,1>>>(100000);
        cudaEventRecord(a); read_k<<<768,256>>>(d,n,sink); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms,a,b); t.push_back(ms*1e3); }
      nvtxRangePop(); std::sort(t.begin(),t.end());
      printf("%-10s events median %8.3f us\n", lab, (t[99]+t[100])/2);
    }
  }
}
