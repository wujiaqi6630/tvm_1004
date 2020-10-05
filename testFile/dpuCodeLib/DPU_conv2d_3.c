void DPU_conv2d_3_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,2:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,2,4:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 27; ++oh) {
        #pragma loop_split(ow,2,4:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 27; ++ow) {
          compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = 0.000000;
          #pragma reduction(ic,compute,+)
          #pragma loop_split(ic,2,3:blockIdx.y,1:local)
          for (int ic = 0; ic < 96; ++ic) {
            for (int kh = 0; kh < 5; ++kh) {
              for (int kw = 0; kw < 5; ++kw) {
                compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = (compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] + (((((2 <= (oh + kh)) && ((oh + kh) < 29)) && (2 <= (ow + kw))) && ((ow + kw) < 29)) ? (placeholder[(((((((on * 69984) + (ic * 729)) + (oh * 27)) + (kh * 27)) + ow) + kw) - 56)] * placeholder1[((((oc * 1200) + (ic * 25)) + (kh * 5)) + kw)]) : 0.000000));
              }
            }
          }
        }
      }
    }
  }
}

