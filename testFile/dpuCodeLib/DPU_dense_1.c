void DPU_dense_1_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma loop_split(oy,3,*:blockIdx.z,4:threadIdx.z,8:local)
  #pragma unroll
  for (int oy = 0; oy < 32; ++oy) {
    #pragma SIMD
    #pragma loop_split(ox,3,*:blockIdx.y,4:threadIdx.y,8:local)
    #pragma unroll
    for (int ox = 0; ox < 4096; ++ox) {
      compute[((oy * 4096) + ox)] = 0.000000;
      #pragma reduction(ok,compute,+)
      #pragma loop_split(oh,2,4:blockIdx.x,1:local)
      for (int ok = 0; ok < 4096; ++ok) {
        compute[((oy * 4096) + ox)] = (compute[((oy * 4096) + ox)] + (placeholder[((oy * 4096) + ok)] * placeholder1[((ok * 4096) + ox)]));
      }
    }
  }
}

