void DPU_conv2d_4_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,8:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 96; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,2:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 55; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,2:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 55; ++ow) {
          compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = 0.000000;
          for (int ic = 0; ic < 3; ++ic) {
            for (int kh = 0; kh < 11; ++kh) {
              for (int kw = 0; kw < 11; ++kw) {
                compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = (compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] + (placeholder[((((((on * 154587) + (ic * 51529)) + (oh * 908)) + (kh * 227)) + (ow * 4)) + kw)] * placeholder1[((((oc * 363) + (ic * 121)) + (kh * 11)) + kw)]));
              }
            }
          }
        }
      }
    }
  }
}

