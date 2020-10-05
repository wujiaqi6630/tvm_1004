void DPU_lrn_sqr_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,16:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,4:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 27; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,4:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 27; ++ow) {
          compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = 0.000000;
          for (int ok = 0; ok < 5; ++ok) {
            compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = (compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] + ((2 <= oc) ? (placeholder[((((((on * 186624) + (oc * 729)) + (ok * 729)) + (oh * 27)) + ow) - 1458)] * placeholder[((((((on * 186624) + (oc * 729)) + (ok * 729)) + (oh * 27)) + ow) - 1458)]) : 0.000000));
          }
        }
      }
    }
  }
}

