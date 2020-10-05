void DPU_dense_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma loop_split(oy,3,*:blockIdx.z,16:threadIdx.z,2:local)
  #pragma unroll
  for (int oy = 0; oy < 32; ++oy) {
    #pragma SIMD
    #pragma loop_split(ox,3,*:blockIdx.y,4:threadIdx.y,4:local)
    #pragma unroll
    for (int ox = 0; ox < 1000; ++ox) {
      compute[((oy * 1000) + ox)] = 0.000000;
      #pragma reduction(ok,compute,+)
      #pragma loop_split(oh,2,2:blockIdx.x,1:local)
      for (int ok = 0; ok < 4096; ++ok) {
        compute[((oy * 1000) + ox)] = (compute[((oy * 1000) + ox)] + (placeholder[((oy * 4096) + ok)] * placeholder1[((ok * 4096) + ox)]));
      }
    }
  }
}

