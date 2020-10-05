void DPU_lrn_pow_1_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,4:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 96; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,8:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 55; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,8:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 55; ++ow) {
          compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = powf(((0.000020 * placeholder[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)]) + 1.000000), 0.750000);
        }
      }
    }
  }
}

