void DPU_softmaxSum_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    compute[on] = 0.000000;
    #pragma reduction(ok,compute,+)
    #pragma loop_split(ok,2,1:blockIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 1000; ++ok) {
      compute[on] = (compute[on] + expf((placeholder1[((on * 1000) + ok)] - placeholder[(on * 1000)])));
    }
  }
}

