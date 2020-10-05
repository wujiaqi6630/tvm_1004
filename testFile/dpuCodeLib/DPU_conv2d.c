void DPU_conv2d_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,16:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,2,2:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 13; ++oh) {
        #pragma loop_split(ow,2,2:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 13; ++ow) {
          compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] = 0.000000;
          #pragma reduction(ic,compute,+)
          #pragma loop_split(ic,2,4:blockIdx.y,1:local)
          for (int ic = 0; ic < 384; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
              for (int kw = 0; kw < 3; ++kw) {
                compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] = (compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] + (((((1 <= (oh + kh)) && ((oh + kh) < 14)) && (1 <= (ow + kw))) && ((ow + kw) < 14)) ? (placeholder[(((((((on * 64896) + (ic * 169)) + (oh * 13)) + (kh * 13)) + ow) + kw) - 14)] * placeholder1[((((oc * 1728) + (ic * 9)) + (kh * 3)) + kw)]) : 0.000000));
              }
            }
          }
        }
      }
    }
  }
}

