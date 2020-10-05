void DPU_softmaxMax_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    compute[on] = -340282346638528859811704183484516925440.000000;
    #pragma reduction(ok,compute,fmax)
    #pragma loop_split(ok,2,1:blockIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 1000; ++ok) {
      float _1 = compute[on];
      float _2 = placeholder[((on * 1000) + ok)];
      compute[on] = ((_1) > (_2) ? (_1) : (_2));
    }
  }
}

