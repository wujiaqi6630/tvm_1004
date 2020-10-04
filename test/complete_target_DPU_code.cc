#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
extern void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* compute = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
   float pad_temp[216];
  for (int32_t i0 = 0; i0 < 2; ++i0) {
    for (int32_t i1 = 0; i1 < 3; ++i1) {
      for (int32_t i2 = 0; i2 < 6; ++i2) {
        for (int32_t i3 = 0; i3 < 6; ++i3) {
          pad_temp[((((i0 * 108) + (i1 * 36)) + (i2 * 6)) + i3)] = (((((1 <= i2) && (i2 < 5)) && (1 <= i3)) && (i3 < 5)) ? placeholder[(((((i0 * 48) + (i1 * 16)) + (i2 * 4)) + i3) - 5)] : 0);
        }
      }
    }
  }
  for (int32_t bh = 0; bh < 2; ++bh) {
    for (int32_t oc = 0; oc < 2; ++oc) {
      for (int32_t oh = 0; oh < 3; ++oh) {
        for (int32_t ow = 0; ow < 3; ++ow) {
          compute[((((bh * 18) + (oc * 9)) + (oh * 3)) + ow)] = 0;
          for (int32_t ic = 0; ic < 3; ++ic) {
            for (int32_t kh = 0; kh < 2; ++kh) {
              for (int32_t kw = 0; kw < 2; ++kw) {
                compute[((((bh * 18) + (oc * 9)) + (oh * 3)) + ow)] = (compute[((((bh * 18) + (oc * 9)) + (oh * 3)) + ow)] + (pad_temp[((((((bh * 108) + (ic * 36)) + (oh * 12)) + (kh * 6)) + (ow * 2)) + kw)] * placeholder1[((((oc * 12) + (ic * 4)) + (kh * 2)) + kw)]));
              }
            }
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
TVM_DLL int32_t fused_nn_softmax( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
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
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
     float tensor1[1];
     float tensor2[10];
     float tensor3[1];
    tensor1[0] = -3.40282e+38;
    for (int32_t k1 = 0; k1 < 10; ++k1) {
      float _1 = tensor1[0];
      float _2 = placeholder[((ax0 * 10) + k1)];
      tensor1[0] = ((_1) > (_2) ? (_1) : (_2));
    }
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      tensor2[ax1] = expf((placeholder[((ax0 * 10) + ax1)] - tensor1[0]));
    }
    tensor3[0] = 0;
    for (int32_t k2 = 0; k2 < 10; ++k2) {
      tensor3[0] = (tensor3[0] + tensor2[k2]);
    }
    for (int32_t ax11 = 0; ax11 < 10; ++ax11) {
      tensor[((ax0 * 10) + ax11)] = (tensor2[ax11] / tensor3[0]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_nn_relu( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  float* T_relu = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
   float compute[20];
  for (int32_t y_outer_x_outer_fused = 0; y_outer_x_outer_fused < 10; ++y_outer_x_outer_fused) {
     float compute1[16];
    (( float8*)(compute1 + 0))[0] = ((float8)(0, 0, 0, 0, 0, 0, 0, 0));
    (( float8*)(compute1 + 8))[0] = ((float8)(0, 0, 0, 0, 0, 0, 0, 0));
    (( float8*)(compute1 + 0))[0] = ((( float8*)(compute1 + 0))[0] + ((( float8*)(placeholder + 0))[0] * (( float8*)(placeholder1 + (y_outer_x_outer_fused * 8)))[0]));
    (( float8*)(compute1 + 8))[0] = ((( float8*)(compute1 + 8))[0] + ((( float8*)(placeholder + 8))[0] * (( float8*)(placeholder1 + (y_outer_x_outer_fused * 8)))[0]));
    for (int32_t y_inner = 0; y_inner < 2; ++y_inner) {
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = 0;
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[(y_inner * 8)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 1)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 2)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 3)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 4)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 5)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 6)]);
      compute[((y_inner * 10) + y_outer_x_outer_fused)] = (compute[((y_inner * 10) + y_outer_x_outer_fused)] + compute1[((y_inner * 8) + 7)]);
    }
  }
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      float _1 = compute[((ax0 * 10) + ax1)];
      T_relu[((ax0 * 10) + ax1)] = ((_1) > (0) ? (_1) : (0));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
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
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 4; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 2; ++ax3) {
        tensor[(((ax0_ax1_fused * 4) + (ax2 * 2)) + ax3)] = -3.40282e+38;
        for (int32_t rv = 0; rv < 2; ++rv) {
          for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
            float _1 = tensor[(((ax0_ax1_fused * 4) + (ax2 * 2)) + ax3)];
            float _2 = ((1 <= ((ax2 * 2) + rv)) && (1 <= ((ax3 * 2) + rv1))) ? placeholder[((((((ax0_ax1_fused * 9) + (ax2 * 6)) + (rv * 3)) + (ax3 * 2)) + rv1) - 4)] : -3.40282e+38;
            tensor[(((ax0_ax1_fused * 4) + (ax2 * 2)) + ax3)] = ((_1) > (_2) ? (_1) : (_2));
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
TVM_DLL int32_t fused_nn_batch_flatten( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
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
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
      tensor[((ax0 * 8) + ax1)] = placeholder[((ax0 * 8) + ax1)];
    }
  }
  return 0;
}

