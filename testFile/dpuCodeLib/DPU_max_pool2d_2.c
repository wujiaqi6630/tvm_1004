void DPU_max_pool2d_2_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,4:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 96; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,4:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 27; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,4:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 27; ++ow) {
          compute[((((on * 69984) + (oc * 729)) + (oh * 27)) + ow)] = -340282346638528859811704183484516925440.000000;
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              float _1 = compute[((((on * 69984) + (oc * 729)) + (oh * 27)) + ow)];
              float _2 = placeholder[((((((on * 290400) + (oc * 3025)) + (oh * 110)) + (kh * 55)) + (ow * 2)) + kw)];
              compute[((((on * 69984) + (oc * 729)) + (oh * 27)) + ow)] = ((_1) > (_2) ? (_1) : (_2));
            }
          }
        }
      }
    }
  }
}

