#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
extern void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_nn_bias_add_nn_relu_4(int fused_nn_conv2d_nn_bias_add_nn_relu_4( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1161600, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  #pragma SIMD
  #pragma unroll
  #pragma loop_split(oc,3,*:blockIdx_z,4:threadIdx_z,1:local)
  #pragma unroll
  for (int32_t oc = 0; oc < 96; ++oc) {
    #pragma loop_split(oh,3,*:blockIdx_y,4:threadIdx_y,7:local)
    #pragma unroll
    for (int32_t oh = 0; oh < 55; ++oh) {
      #pragma loop_split(ow,3,*:blockIdx_x,4:threadIdx_x,7:local)
      #pragma unroll
      for (int32_t ow = 0; ow < 55; ++ow) {
        (( float*)compute)[(((oc * 3025) + (oh * 55)) + ow)] = 0;
        for (int32_t ic = 0; ic < 3; ++ic) {
          for (int32_t kh = 0; kh < 11; ++kh) {
            for (int32_t kw = 0; kw < 11; ++kw) {
              (( float*)compute)[(((oc * 3025) + (oh * 55)) + ow)] = ((( float*)compute)[(((oc * 3025) + (oh * 55)) + ow)] + (placeholder[(((((ic * 51529) + (oh * 908)) + (kh * 227)) + (ow * 4)) + kw)] * placeholder1[((((oc * 363) + (ic * 121)) + (kh * 11)) + kw)]));
            }
          }
        }
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 96; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 55; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 55; ++ax3) {
        float _1 = (( float*)compute)[(((ax1 * 3025) + (ax2 * 55)) + ax3)] + placeholder2[ax1];
        T_relu[(((ax1 * 3025) + (ax2 * 55)) + ax3)] = ((_1) > (0) ? (_1) : (0));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d_2(int fused_nn_max_pool2d_2( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* compute = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t n_oc_fused = 0; n_oc_fused < 96; ++n_oc_fused) {
    for (int32_t oh = 0; oh < 27; ++oh) {
      for (int32_t ow = 0; ow < 27; ++ow) {
        compute[(((n_oc_fused * 729) + (oh * 27)) + ow)] = -3.40282e+38;
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            float _1 = compute[(((n_oc_fused * 729) + (oh * 27)) + ow)];
            float _2 = placeholder[(((((n_oc_fused * 3025) + (oh * 110)) + (kh * 55)) + (ow * 2)) + kw)];
            compute[(((n_oc_fused * 729) + (oh * 27)) + ow)] = ((_1) > (_2) ? (_1) : (_2));
          }
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_nn_bias_add_nn_relu_3(int fused_nn_conv2d_nn_bias_add_nn_relu_3( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)369024, 2, 32);
  if (pad_temp == NULL) {
    return -1;
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)746496, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  void* T_expand_dims = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
  if (T_expand_dims == NULL) {
    return -1;
  }
  for (int32_t i1 = 0; i1 < 96; ++i1) {
    for (int32_t i2 = 0; i2 < 31; ++i2) {
      for (int32_t i3 = 0; i3 < 31; ++i3) {
        (( float*)pad_temp)[(((i1 * 961) + (i2 * 31)) + i3)] = (((((2 <= i2) && (i2 < 29)) && (2 <= i3)) && (i3 < 29)) ? placeholder[((((i1 * 729) + (i2 * 27)) + i3) - 56)] : 0);
      }
    }
  }
  for (int32_t ff = 0; ff < 256; ++ff) {
    for (int32_t yy = 0; yy < 27; ++yy) {
      for (int32_t xx = 0; xx < 27; ++xx) {
        (( float*)compute)[(((ff * 729) + (yy * 27)) + xx)] = 0;
        for (int32_t rc = 0; rc < 48; ++rc) {
          for (int32_t ry = 0; ry < 5; ++ry) {
            for (int32_t rx = 0; rx < 5; ++rx) {
              (( float*)compute)[(((ff * 729) + (yy * 27)) + xx)] = ((( float*)compute)[(((ff * 729) + (yy * 27)) + xx)] + ((( float*)pad_temp)[(((((((ff >> 7) * 46128) + (rc * 961)) + (yy * 31)) + (ry * 31)) + xx) + rx)] * placeholder1[((((ff * 1200) + (rc * 25)) + (ry * 5)) + rx)]));
            }
          }
        }
      }
    }
  }
  for (int32_t ax0 = 0; ax0 < 256; ++ax0) {
    (( float*)T_expand_dims)[ax0] = placeholder2[ax0];
  }
  for (int32_t ax1 = 0; ax1 < 256; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 27; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 27; ++ax3) {
        (( float*)compute)[(((ax1 * 729) + (ax2 * 27)) + ax3)] = ((( float*)compute)[(((ax1 * 729) + (ax2 * 27)) + ax3)] + (( float*)T_expand_dims)[ax1]);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 256; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 27; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 27; ++ax31) {
        float _1 = (( float*)compute)[(((ax11 * 729) + (ax21 * 27)) + ax31)];
        T_relu[(((ax11 * 729) + (ax21 * 27)) + ax31)] = ((_1) > (0) ? (_1) : (0));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_expand_dims) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_nn_bias_add(int fused_nn_dense_nn_bias_add( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4000, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  for (int32_t y_outer_x_outer_fused = 0; y_outer_x_outer_fused < 1000; ++y_outer_x_outer_fused) {
     float compute1[16];
    (( float16*)(compute1 + 0))[0] = ((float16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    for (int32_t k = 0; k < 256; ++k) {
      (( float16*)(compute1 + 0))[0] = ((( float16*)(compute1 + 0))[0] + ((( float16*)(placeholder + (k * 16)))[0] * (( float16*)(placeholder1 + ((y_outer_x_outer_fused * 4096) + (k * 16))))[0]));
    }
    (( float*)compute)[y_outer_x_outer_fused] = 0;
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[0]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[1]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[2]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[3]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[4]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[5]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[6]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[7]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[8]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[9]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[10]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[11]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[12]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[13]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[14]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[15]);
  }
  for (int32_t ax1 = 0; ax1 < 1000; ++ax1) {
    T_add[ax1] = ((( float*)compute)[ax1] + placeholder2[ax1]);
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_nn_bias_add_nn_relu_1(int fused_nn_conv2d_nn_bias_add_nn_relu_1( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)345600, 2, 32);
  if (pad_temp == NULL) {
    return -1;
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)259584, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  void* T_expand_dims = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1536, 2, 32);
  if (T_expand_dims == NULL) {
    return -1;
  }
  for (int32_t i1 = 0; i1 < 384; ++i1) {
    for (int32_t i2 = 0; i2 < 15; ++i2) {
      for (int32_t i3 = 0; i3 < 15; ++i3) {
        (( float*)pad_temp)[(((i1 * 225) + (i2 * 15)) + i3)] = (((((1 <= i2) && (i2 < 14)) && (1 <= i3)) && (i3 < 14)) ? placeholder[((((i1 * 169) + (i2 * 13)) + i3) - 14)] : 0);
      }
    }
  }
  for (int32_t ff = 0; ff < 384; ++ff) {
    for (int32_t yy = 0; yy < 13; ++yy) {
      for (int32_t xx = 0; xx < 13; ++xx) {
        (( float*)compute)[(((ff * 169) + (yy * 13)) + xx)] = 0;
        for (int32_t rc = 0; rc < 192; ++rc) {
          for (int32_t ry = 0; ry < 3; ++ry) {
            for (int32_t rx = 0; rx < 3; ++rx) {
              (( float*)compute)[(((ff * 169) + (yy * 13)) + xx)] = ((( float*)compute)[(((ff * 169) + (yy * 13)) + xx)] + ((( float*)pad_temp)[(((((((ff / 192) * 43200) + (rc * 225)) + (yy * 15)) + (ry * 15)) + xx) + rx)] * placeholder1[((((ff * 1728) + (rc * 9)) + (ry * 3)) + rx)]));
            }
          }
        }
      }
    }
  }
  for (int32_t ax0 = 0; ax0 < 384; ++ax0) {
    (( float*)T_expand_dims)[ax0] = placeholder2[ax0];
  }
  for (int32_t ax1 = 0; ax1 < 384; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 13; ++ax3) {
        (( float*)compute)[(((ax1 * 169) + (ax2 * 13)) + ax3)] = ((( float*)compute)[(((ax1 * 169) + (ax2 * 13)) + ax3)] + (( float*)T_expand_dims)[ax1]);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 384; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 13; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 13; ++ax31) {
        float _1 = (( float*)compute)[(((ax11 * 169) + (ax21 * 13)) + ax31)];
        T_relu[(((ax11 * 169) + (ax21 * 13)) + ax31)] = ((_1) > (0) ? (_1) : (0));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_expand_dims) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_nn_bias_add_nn_relu_2(int fused_nn_conv2d_nn_bias_add_nn_relu_2( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)259584, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  #pragma SIMD
  #pragma unroll
  #pragma loop_split(oc,3,*:blockIdx_z,4:threadIdx_z,1:local)
  #pragma unroll
  for (int32_t oc = 0; oc < 384; ++oc) {
    #pragma loop_split(oh,3,*:blockIdx_y,4:threadIdx_y,7:local)
    #pragma unroll
    for (int32_t oh = 0; oh < 13; ++oh) {
      #pragma loop_split(ow,3,*:blockIdx_x,4:threadIdx_x,7:local)
      #pragma unroll
      for (int32_t ow = 0; ow < 13; ++ow) {
        (( float*)compute)[(((oc * 169) + (oh * 13)) + ow)] = 0;
        for (int32_t ic = 0; ic < 256; ++ic) {
          for (int32_t kh = 0; kh < 3; ++kh) {
            for (int32_t kw = 0; kw < 3; ++kw) {
              (( float*)compute)[(((oc * 169) + (oh * 13)) + ow)] = ((( float*)compute)[(((oc * 169) + (oh * 13)) + ow)] + (((((1 <= (oh + kh)) && ((oh + kh) < 14)) && (1 <= (ow + kw))) && ((ow + kw) < 14)) ? (placeholder[((((((ic * 169) + (oh * 13)) + (kh * 13)) + ow) + kw) - 14)] * placeholder1[((((oc * 2304) + (ic * 9)) + (kh * 3)) + kw)]) : 0));
            }
          }
        }
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 384; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 13; ++ax3) {
        float _1 = (( float*)compute)[(((ax1 * 169) + (ax2 * 13)) + ax3)] + placeholder2[ax1];
        T_relu[(((ax1 * 169) + (ax2 * 13)) + ax3)] = ((_1) > (0) ? (_1) : (0));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_softmax(int fused_nn_softmax( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* tensor = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
   float tensor1[1];
  void* tensor2 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4000, 2, 32);
  if (tensor2 == NULL) {
    return -1;
  }
   float tensor3[1];
  tensor1[0] = -3.40282e+38;
  for (int32_t k1 = 0; k1 < 1000; ++k1) {
    float _1 = tensor1[0];
    float _2 = placeholder[k1];
    tensor1[0] = ((_1) > (_2) ? (_1) : (_2));
  }
  for (int32_t ax1 = 0; ax1 < 1000; ++ax1) {
    (( float*)tensor2)[ax1] = expf((placeholder[ax1] - tensor1[0]));
  }
  tensor3[0] = 0;
  for (int32_t k2 = 0; k2 < 1000; ++k2) {
    tensor3[0] = (tensor3[0] + (( float*)tensor2)[k2]);
  }
  for (int32_t ax11 = 0; ax11 < 1000; ++ax11) {
    tensor[ax11] = ((( float*)tensor2)[ax11] / tensor3[0]);
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor2) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_lrn_1(int fused_nn_lrn_1( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* T_divide = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* pad_data = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1210000, 2, 32);
  if (pad_data == NULL) {
    return -1;
  }
  void* tensor = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1161600, 2, 32);
  if (tensor == NULL) {
    return -1;
  }
  for (int32_t ax1 = 0; ax1 < 100; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 55; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 55; ++ax3) {
        (( float*)pad_data)[(((ax1 * 3025) + (ax2 * 55)) + ax3)] = (((2 <= ax1) && (ax1 < 98)) ? placeholder[((((ax1 * 3025) + (ax2 * 55)) + ax3) - 6050)] : 0);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 96; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 55; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 55; ++ax31) {
        (( float*)tensor)[(((ax11 * 3025) + (ax21 * 55)) + ax31)] = 0;
        for (int32_t rxs = 0; rxs < 5; ++rxs) {
          (( float*)tensor)[(((ax11 * 3025) + (ax21 * 55)) + ax31)] = ((( float*)tensor)[(((ax11 * 3025) + (ax21 * 55)) + ax31)] + ((( float*)pad_data)[((((ax11 * 3025) + (rxs * 3025)) + (ax21 * 55)) + ax31)] * (( float*)pad_data)[((((ax11 * 3025) + (rxs * 3025)) + (ax21 * 55)) + ax31)]));
        }
      }
    }
  }
  for (int32_t ax12 = 0; ax12 < 96; ++ax12) {
    for (int32_t ax22 = 0; ax22 < 55; ++ax22) {
      for (int32_t ax32 = 0; ax32 < 55; ++ax32) {
        (( float*)tensor)[(((ax12 * 3025) + (ax22 * 55)) + ax32)] = powf((1 + ((0.0001 * (( float*)tensor)[(((ax12 * 3025) + (ax22 * 55)) + ax32)]) * 0.2)), 0.75);
      }
    }
  }
  for (int32_t ax13 = 0; ax13 < 96; ++ax13) {
    for (int32_t ax23 = 0; ax23 < 55; ++ax23) {
      for (int32_t ax33 = 0; ax33 < 55; ++ax33) {
        T_divide[(((ax13 * 3025) + (ax23 * 55)) + ax33)] = (placeholder[(((ax13 * 3025) + (ax23 * 55)) + ax33)] / (( float*)tensor)[(((ax13 * 3025) + (ax23 * 55)) + ax33)]);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_data) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d_1(int fused_nn_max_pool2d_1( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* compute = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t n_oc_fused = 0; n_oc_fused < 256; ++n_oc_fused) {
    for (int32_t oh = 0; oh < 13; ++oh) {
      for (int32_t ow = 0; ow < 13; ++ow) {
        compute[(((n_oc_fused * 169) + (oh * 13)) + ow)] = -3.40282e+38;
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            float _1 = compute[(((n_oc_fused * 169) + (oh * 13)) + ow)];
            float _2 = placeholder[(((((n_oc_fused * 729) + (oh * 54)) + (kh * 27)) + (ow * 2)) + kw)];
            compute[(((n_oc_fused * 169) + (oh * 13)) + ow)] = ((_1) > (_2) ? (_1) : (_2));
          }
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_nn_bias_add_nn_relu_1(int fused_nn_dense_nn_bias_add_nn_relu_1( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)16384, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  for (int32_t y_outer_x_outer_fused = 0; y_outer_x_outer_fused < 4096; ++y_outer_x_outer_fused) {
     float compute1[16];
    (( float16*)(compute1 + 0))[0] = ((float16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    for (int32_t k = 0; k < 576; ++k) {
      (( float16*)(compute1 + 0))[0] = ((( float16*)(compute1 + 0))[0] + ((( float16*)(placeholder + (k * 16)))[0] * (( float16*)(placeholder1 + ((y_outer_x_outer_fused * 9216) + (k * 16))))[0]));
    }
    (( float*)compute)[y_outer_x_outer_fused] = 0;
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[0]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[1]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[2]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[3]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[4]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[5]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[6]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[7]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[8]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[9]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[10]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[11]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[12]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[13]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[14]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[15]);
  }
  for (int32_t ax1 = 0; ax1 < 4096; ++ax1) {
    float _1 = (( float*)compute)[ax1] + placeholder2[ax1];
    T_relu[ax1] = ((_1) > (0) ? (_1) : (0));
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_reshape(int fused_reshape( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* T_reshape = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax1 = 0; ax1 < 4096; ++ax1) {
    T_reshape[ax1] = placeholder[ax1];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_nn_bias_add_nn_relu(int fused_nn_dense_nn_bias_add_nn_relu( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)16384, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  for (int32_t y_outer_x_outer_fused = 0; y_outer_x_outer_fused < 4096; ++y_outer_x_outer_fused) {
     float compute1[16];
    (( float16*)(compute1 + 0))[0] = ((float16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    for (int32_t k = 0; k < 256; ++k) {
      (( float16*)(compute1 + 0))[0] = ((( float16*)(compute1 + 0))[0] + ((( float16*)(placeholder + (k * 16)))[0] * (( float16*)(placeholder1 + ((y_outer_x_outer_fused * 4096) + (k * 16))))[0]));
    }
    (( float*)compute)[y_outer_x_outer_fused] = 0;
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[0]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[1]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[2]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[3]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[4]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[5]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[6]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[7]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[8]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[9]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[10]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[11]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[12]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[13]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[14]);
    (( float*)compute)[y_outer_x_outer_fused] = ((( float*)compute)[y_outer_x_outer_fused] + compute1[15]);
  }
  for (int32_t ax1 = 0; ax1 < 4096; ++ax1) {
    float _1 = (( float*)compute)[ax1] + placeholder2[ax1];
    T_relu[ax1] = ((_1) > (0) ? (_1) : (0));
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d(int fused_nn_max_pool2d( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* compute = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t n_oc_fused = 0; n_oc_fused < 256; ++n_oc_fused) {
    for (int32_t oh = 0; oh < 6; ++oh) {
      for (int32_t ow = 0; ow < 6; ++ow) {
        compute[(((n_oc_fused * 36) + (oh * 6)) + ow)] = -3.40282e+38;
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            float _1 = compute[(((n_oc_fused * 36) + (oh * 6)) + ow)];
            float _2 = placeholder[(((((n_oc_fused * 169) + (oh * 26)) + (kh * 13)) + (ow * 2)) + kw)];
            compute[(((n_oc_fused * 36) + (oh * 6)) + ow)] = ((_1) > (_2) ? (_1) : (_2));
          }
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_lrn(int fused_nn_lrn( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* T_divide = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* pad_data = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)758160, 2, 32);
  if (pad_data == NULL) {
    return -1;
  }
  void* tensor = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)746496, 2, 32);
  if (tensor == NULL) {
    return -1;
  }
  for (int32_t ax1 = 0; ax1 < 260; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 27; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 27; ++ax3) {
        (( float*)pad_data)[(((ax1 * 729) + (ax2 * 27)) + ax3)] = (((2 <= ax1) && (ax1 < 258)) ? placeholder[((((ax1 * 729) + (ax2 * 27)) + ax3) - 1458)] : 0);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 256; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 27; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 27; ++ax31) {
        (( float*)tensor)[(((ax11 * 729) + (ax21 * 27)) + ax31)] = 0;
        for (int32_t rxs = 0; rxs < 5; ++rxs) {
          (( float*)tensor)[(((ax11 * 729) + (ax21 * 27)) + ax31)] = ((( float*)tensor)[(((ax11 * 729) + (ax21 * 27)) + ax31)] + ((( float*)pad_data)[((((ax11 * 729) + (rxs * 729)) + (ax21 * 27)) + ax31)] * (( float*)pad_data)[((((ax11 * 729) + (rxs * 729)) + (ax21 * 27)) + ax31)]));
        }
      }
    }
  }
  for (int32_t ax12 = 0; ax12 < 256; ++ax12) {
    for (int32_t ax22 = 0; ax22 < 27; ++ax22) {
      for (int32_t ax32 = 0; ax32 < 27; ++ax32) {
        (( float*)tensor)[(((ax12 * 729) + (ax22 * 27)) + ax32)] = powf((1 + ((0.0001 * (( float*)tensor)[(((ax12 * 729) + (ax22 * 27)) + ax32)]) * 0.2)), 0.75);
      }
    }
  }
  for (int32_t ax13 = 0; ax13 < 256; ++ax13) {
    for (int32_t ax23 = 0; ax23 < 27; ++ax23) {
      for (int32_t ax33 = 0; ax33 < 27; ++ax33) {
        T_divide[(((ax13 * 729) + (ax23 * 27)) + ax33)] = (placeholder[(((ax13 * 729) + (ax23 * 27)) + ax33)] / (( float*)tensor)[(((ax13 * 729) + (ax23 * 27)) + ax33)]);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_data) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_reshape_1(int fused_reshape_1( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* T_reshape = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax1 = 0; ax1 < 9216; ++ax1) {
    T_reshape[ax1] = placeholder[ax1];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_nn_bias_add_nn_relu(int fused_nn_conv2d_nn_bias_add_nn_relu( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)345600, 2, 32);
  if (pad_temp == NULL) {
    return -1;
  }
  void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)173056, 2, 32);
  if (compute == NULL) {
    return -1;
  }
  void* T_expand_dims = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
  if (T_expand_dims == NULL) {
    return -1;
  }
  for (int32_t i1 = 0; i1 < 384; ++i1) {
    for (int32_t i2 = 0; i2 < 15; ++i2) {
      for (int32_t i3 = 0; i3 < 15; ++i3) {
        (( float*)pad_temp)[(((i1 * 225) + (i2 * 15)) + i3)] = (((((1 <= i2) && (i2 < 14)) && (1 <= i3)) && (i3 < 14)) ? placeholder[((((i1 * 169) + (i2 * 13)) + i3) - 14)] : 0);
      }
    }
  }
  for (int32_t ff = 0; ff < 256; ++ff) {
    for (int32_t yy = 0; yy < 13; ++yy) {
      for (int32_t xx = 0; xx < 13; ++xx) {
        (( float*)compute)[(((ff * 169) + (yy * 13)) + xx)] = 0;
        for (int32_t rc = 0; rc < 192; ++rc) {
          for (int32_t ry = 0; ry < 3; ++ry) {
            for (int32_t rx = 0; rx < 3; ++rx) {
              (( float*)compute)[(((ff * 169) + (yy * 13)) + xx)] = ((( float*)compute)[(((ff * 169) + (yy * 13)) + xx)] + ((( float*)pad_temp)[(((((((ff >> 7) * 43200) + (rc * 225)) + (yy * 15)) + (ry * 15)) + xx) + rx)] * placeholder1[((((ff * 1728) + (rc * 9)) + (ry * 3)) + rx)]));
            }
          }
        }
      }
    }
  }
  for (int32_t ax0 = 0; ax0 < 256; ++ax0) {
    (( float*)T_expand_dims)[ax0] = placeholder2[ax0];
  }
  for (int32_t ax1 = 0; ax1 < 256; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 13; ++ax3) {
        (( float*)compute)[(((ax1 * 169) + (ax2 * 13)) + ax3)] = ((( float*)compute)[(((ax1 * 169) + (ax2 * 13)) + ax3)] + (( float*)T_expand_dims)[ax1]);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 256; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 13; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 13; ++ax31) {
        float _1 = (( float*)compute)[(((ax11 * 169) + (ax21 * 13)) + ax31)];
        T_relu[(((ax11 * 169) + (ax21 * 13)) + ax31)] = ((_1) > (0) ? (_1) : (0));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_expand_dims) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}


