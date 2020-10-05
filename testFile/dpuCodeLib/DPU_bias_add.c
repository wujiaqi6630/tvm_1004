void DPU_bias_add_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(ok,3,*:blockIdx.z,1:threadIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 1000; ++ok) {
      compute[((on * 1000) + ok)] = (placeholder[((on * 1000) + ok)] + placeholder1[ok]);
    }
  }
}

