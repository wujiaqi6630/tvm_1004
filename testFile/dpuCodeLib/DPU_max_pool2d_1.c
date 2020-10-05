void DPU_max_pool2d_1_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,16:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,2:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 13; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,2:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 13; ++ow) {
          compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] = -340282346638528859811704183484516925440.000000;
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              float _1 = compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)];
              float _2 = placeholder[((((((on * 186624) + (oc * 729)) + (oh * 54)) + (kh * 27)) + (ow * 2)) + kw)];
              compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] = ((_1) > (_2) ? (_1) : (_2));
            }
          }
        }
      }
    }
  }
}

