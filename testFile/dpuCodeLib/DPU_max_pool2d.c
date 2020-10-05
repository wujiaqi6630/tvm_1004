void DPU_max_pool2d_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,64:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,4:threadIdx.y,1:local)
      #pragma unroll
      for (int oh = 0; oh < 6; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,4:threadIdx.x,1:local)
        #pragma unroll
        for (int ow = 0; ow < 6; ++ow) {
          compute[((((on * 9216) + (oc * 36)) + (oh * 6)) + ow)] = -340282346638528859811704183484516925440.000000;
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              float _1 = compute[((((on * 9216) + (oc * 36)) + (oh * 6)) + ow)];
              float _2 = placeholder[((((((on * 43264) + (oc * 169)) + (oh * 26)) + (kh * 13)) + (ow * 2)) + kw)];
              compute[((((on * 9216) + (oc * 36)) + (oh * 6)) + ow)] = ((_1) > (_2) ? (_1) : (_2));
            }
          }
        }
      }
    }
  }
}

