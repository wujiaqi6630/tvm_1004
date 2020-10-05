---------------------DEVICE_CODE------------------------


void DPU_bias_add_5_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,1:threadIdx.z,2:local)
    #pragma unroll
    for (int oc = 0; oc < 96; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,8:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 55; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,8:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 55; ++ow) {
          compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = (placeholder[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] + placeholder1[oc]);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_bias_add_4_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,4:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,4:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 27; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,4:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 27; ++ow) {
          compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = (placeholder[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] + placeholder1[oc]);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_lrn_div_1_kernel(float* placeholder,  float* placeholder1,  float* compute) {
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
          compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = (placeholder1[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] / placeholder[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)]);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_dense_2_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma loop_split(oy,3,*:blockIdx.z,8:threadIdx.z,4:local)
  #pragma unroll
  for (int oy = 0; oy < 32; ++oy) {
    #pragma SIMD
    #pragma loop_split(ox,3,*:blockIdx.y,8:threadIdx.y,4:local)
    #pragma unroll
    for (int ox = 0; ox < 4096; ++ox) {
      compute[((oy * 4096) + ox)] = 0.000000;
      #pragma reduction(ok,compute,+)
      #pragma loop_split(oh,2,4:blockIdx.x,1:local)
      for (int ok = 0; ok < 9216; ++ok) {
        compute[((oy * 4096) + ox)] = (compute[((oy * 4096) + ox)] + (placeholder[((oy * 9216) + ok)] * placeholder1[((ok * 9216) + ox)]));
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_bias_add_1_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(ok,3,*:blockIdx.z,4:threadIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 4096; ++ok) {
      compute[((on * 4096) + ok)] = (placeholder[((on * 4096) + ok)] + placeholder1[ok]);
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_relu_2_kernel(float* placeholder,  float* compute) {
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
          compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] = ((0.000000 < placeholder[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)]) ? placeholder[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] : 0.000000);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_relu_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(ok,3,*:blockIdx.z,4:threadIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 4096; ++ok) {
      compute[((on * 4096) + ok)] = ((0.000000 < placeholder[((on * 4096) + ok)]) ? placeholder[((on * 4096) + ok)] : 0.000000);
    }
  }
}

---------------------DEVICE_CODE------------------------


void fused_reshape_kernel(float* placeholder,  float* T_reshape) {
  for (int ax0 = 0; ax0 < 32; ++ax0) {
    for (int ax1 = 0; ax1 < 4096; ++ax1) {
      T_reshape[((ax0 * 4096) + ax1)] = placeholder[((ax0 * 4096) + ax1)];
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_dense_1_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma loop_split(oy,3,*:blockIdx.z,4:threadIdx.z,8:local)
  #pragma unroll
  for (int oy = 0; oy < 32; ++oy) {
    #pragma SIMD
    #pragma loop_split(ox,3,*:blockIdx.y,4:threadIdx.y,8:local)
    #pragma unroll
    for (int ox = 0; ox < 4096; ++ox) {
      compute[((oy * 4096) + ox)] = 0.000000;
      #pragma reduction(ok,compute,+)
      #pragma loop_split(oh,2,4:blockIdx.x,1:local)
      for (int ok = 0; ok < 4096; ++ok) {
        compute[((oy * 4096) + ox)] = (compute[((oy * 4096) + ox)] + (placeholder[((oy * 4096) + ok)] * placeholder1[((ok * 4096) + ox)]));
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_relu_3_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,4:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 256; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,4:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 27; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,4:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 27; ++ow) {
          compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = ((0.000000 < placeholder[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)]) ? placeholder[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] : 0.000000);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_softmaxDiv_kernel(float* placeholder,  float* placeholder1,  float* placeholder2,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(ok,3,*:blockIdx.z,1:threadIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 1000; ++ok) {
      compute[((on * 1000) + ok)] = (expf((placeholder2[((on * 1000) + ok)] - placeholder1[(on * 1000)])) / placeholder[(on * 1000)]);
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_softmaxSum_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    compute[on] = 0.000000;
    #pragma reduction(ok,compute,+)
    #pragma loop_split(ok,2,1:blockIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 1000; ++ok) {
      compute[on] = (compute[on] + expf((placeholder1[((on * 1000) + ok)] - placeholder[(on * 1000)])));
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_conv2d_2_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,16:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 384; ++oc) {
      #pragma loop_split(oh,2,2:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 13; ++oh) {
        #pragma loop_split(ow,2,2:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 13; ++ow) {
          compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] = 0.000000;
          #pragma reduction(ic,compute,+)
          #pragma loop_split(ic,2,4:blockIdx.y,1:local)
          for (int ic = 0; ic < 256; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
              for (int kw = 0; kw < 3; ++kw) {
                compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] = (compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] + (((((1 <= (oh + kh)) && ((oh + kh) < 14)) && (1 <= (ow + kw))) && ((ow + kw) < 14)) ? (placeholder[(((((((on * 43264) + (ic * 169)) + (oh * 13)) + (kh * 13)) + ow) + kw) - 14)] * placeholder1[((((oc * 2304) + (ic * 9)) + (kh * 3)) + kw)]) : 0.000000));
              }
            }
          }
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_lrn_div_kernel(float* placeholder,  float* placeholder1,  float* compute) {
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
          compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = (placeholder1[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] / placeholder[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)]);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_relu_4_kernel(float* placeholder,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,1:threadIdx.z,2:local)
    #pragma unroll
    for (int oc = 0; oc < 96; ++oc) {
      #pragma loop_split(oh,3,*:blockIdx.y,8:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 55; ++oh) {
        #pragma loop_split(ow,3,*:blockIdx.x,8:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 55; ++ow) {
          compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = ((0.000000 < placeholder[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)]) ? placeholder[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] : 0.000000);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_bias_add_2_kernel(float* placeholder,  float* placeholder1,  float* compute) {
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
          compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] = (placeholder[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] + placeholder1[oc]);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_conv2d_1_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(oc,3,*:blockIdx.z,16:threadIdx.z,1:local)
    #pragma unroll
    for (int oc = 0; oc < 384; ++oc) {
      #pragma loop_split(oh,2,2:threadIdx.y,7:local)
      #pragma unroll
      for (int oh = 0; oh < 13; ++oh) {
        #pragma loop_split(ow,2,2:threadIdx.x,7:local)
        #pragma unroll
        for (int ow = 0; ow < 13; ++ow) {
          compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] = 0.000000;
          #pragma reduction(ic,compute,+)
          #pragma loop_split(ic,2,4:blockIdx.y,1:local)
          for (int ic = 0; ic < 384; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
              for (int kw = 0; kw < 3; ++kw) {
                compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] = (compute[((((on * 64896) + (oc * 169)) + (oh * 13)) + ow)] + (((((1 <= (oh + kh)) && ((oh + kh) < 14)) && (1 <= (ow + kw))) && ((ow + kw) < 14)) ? (placeholder[(((((((on * 64896) + (ic * 169)) + (oh * 13)) + (kh * 13)) + ow) + kw) - 14)] * placeholder1[((((oc * 1728) + (ic * 9)) + (kh * 3)) + kw)]) : 0.000000));
              }
            }
          }
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_dense_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma loop_split(oy,3,*:blockIdx.z,16:threadIdx.z,2:local)
  #pragma unroll
  for (int oy = 0; oy < 32; ++oy) {
    #pragma SIMD
    #pragma loop_split(ox,3,*:blockIdx.y,4:threadIdx.y,4:local)
    #pragma unroll
    for (int ox = 0; ox < 1000; ++ox) {
      compute[((oy * 1000) + ox)] = 0.000000;
      #pragma reduction(ok,compute,+)
      #pragma loop_split(oh,2,2:blockIdx.x,1:local)
      for (int ok = 0; ok < 4096; ++ok) {
        compute[((oy * 1000) + ox)] = (compute[((oy * 1000) + ox)] + (placeholder[((oy * 4096) + ok)] * placeholder1[((ok * 4096) + ox)]));
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_relu_1_kernel(float* placeholder,  float* compute) {
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
          compute[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] = ((0.000000 < placeholder[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)]) ? placeholder[((((on * 43264) + (oc * 169)) + (oh * 13)) + ow)] : 0.000000);
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void fused_reshape_1_kernel(float* placeholder,  float* T_reshape) {
  for (int ax0 = 0; ax0 < 32; ++ax0) {
    for (int ax1 = 0; ax1 < 9216; ++ax1) {
      T_reshape[((ax0 * 9216) + ax1)] = placeholder[((ax0 * 9216) + ax1)];
    }
  }
}

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


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

---------------------DEVICE_CODE------------------------


void DPU_lrn_sqr_1_kernel(float* placeholder,  float* compute) {
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
          compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = 0.000000;
          for (int ok = 0; ok < 5; ++ok) {
            compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] = (compute[((((on * 290400) + (oc * 3025)) + (oh * 55)) + ow)] + ((2 <= oc) ? (placeholder[((((((on * 290400) + (oc * 3025)) + (ok * 3025)) + (oh * 55)) + ow) - 6050)] * placeholder[((((((on * 290400) + (oc * 3025)) + (ok * 3025)) + (oh * 55)) + ow) - 6050)]) : 0.000000));
          }
        }
      }
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_bias_add_kernel(float* placeholder,  float* placeholder1,  float* compute) {
  #pragma SIMD
  #pragma unroll
  for (int on = 0; on < 32; ++on) {
    #pragma loop_split(ok,3,*:blockIdx.z,1:threadIdx.z,1024:local)
    #pragma unroll
    for (int ok = 0; ok < 1000; ++ok) {
      compute[((on * 1000) + ok)] = (placeholder[((on * 1000) + ok)] + placeholder1[ok]);
    }
  }
}

---------------------DEVICE_CODE------------------------


void DPU_lrn_pow_kernel(float* placeholder,  float* compute) {
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
          compute[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)] = powf(((0.000020 * placeholder[((((on * 186624) + (oc * 729)) + (oh * 27)) + ow)]) + 1.000000), 0.750000);
        }
      }
    }
  }
}


