void DPU_bias_add_3_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,16:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 384; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,2:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 13; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,2:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 13; ++ow) {
          compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] = (placeholder[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] + placeholder1[oc]);
        }
      }
    }
  }
}

