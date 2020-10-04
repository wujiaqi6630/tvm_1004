/* PROGRAM DESIGN METHOD
 * input_data and kernel_data should come from user's program script
 */
//#include "tvm/runtime/c_runtime_api.h"
//#include "tvm/runtime/c_backend_api.h"
//extern void* __tvm_module_ctx = NULL;
//#ifdef __cplusplus
//extern "C"
//#endif
/***modify in codegen_host_c.cc***/

//TVM_DLL int32_t fused_nn_conv2d( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
int fused_nn_conv2d(float* placeholder, float* placeholder1, float* compute) {
  //void* arg0 = (((TVMValue*)args)[0].v_handle);
  //int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  //void* arg1 = (((TVMValue*)args)[1].v_handle);
  //int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  //void* arg2 = (((TVMValue*)args)[2].v_handle);
  //int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  //float* placeholder = (float*)(((TVMArray*)arg0)[0].data);  //palceholder means conv2d layer's input_data
  //int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  //int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  //int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  //int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  //float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data); //palceholder1 means conv2d layer's kernel_data
  //int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  //int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  //float* compute = (float*)(((TVMArray*)arg2)[0].data);      //compute means conv2d layer's output_data
  //int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  //int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  //if (!(arg0_strides == NULL)) {
  //}
  //if (!(arg1_strides == NULL)) {
  //}
  //if (!(arg2_strides == NULL)) {
  //}
   float pad_temp[216];                                      //pad_temp means padded_input_data
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

