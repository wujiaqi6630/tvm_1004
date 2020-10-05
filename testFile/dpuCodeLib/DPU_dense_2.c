void DPU_dense_2_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma loop_split(oy,3,*:blockIdx.z,8:threadIdx.z,4:local)
  #pragma unroll
  for (int oy = 0; oy < 32; ++oy) {
    #pragma SIMD
    #pragma loop_split(ox,3,*:blockIdx.y,8:threadIdx.y,4:local)
    #pragma unroll
    for (int ox = 0; ox < 4096; ++ox) {
      compute[((oy * 4096) + ox)] = 0.000000;
      #pragma reduction(ok,compute,+)
      #pragma loop_split(oh,2,4:blockIdx.x,1:local)
      for (int ok = 0; ok < 9216; ++ok) {
        compute[((oy * 4096) + ox)] = (compute[((oy * 4096) + ox)] + (placeholder[((oy * 9216) + ok)] * placeholder1[((ok * 9216) + ox)]));
      }
    }
  }
}

