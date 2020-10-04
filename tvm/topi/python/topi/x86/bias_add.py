# wjq 2020/09/16

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from ..nn.bias_add import bias_add

from ..util import simplify


@autotvm.register_topi_compute(bias_add, 'cpu', ['direct'])
def _declaration_bias_add (cfg, input, bias, axis=1, out_dtype=None):
    data_ndim = len(input.shape)
    if axis < 0:
        axis = axis + data_ndim
    if data_ndim == 2:
        return bias_add2d_compute(input, bias, axis, out_dtype)
    elif data_ndim == 4:
        return bias_add4d_compute(input, bias, axis, out_dtype)
    raise ValueError("bias_add op's input data shape should be 2 or 4.")


def bias_add2d_compute(input, bias, axis=1, out_dtype=None):
    if out_dtype is None:
        out_dtype = input.dtype

    batch, species = input.shape

    data_ndim = len(input.shape)
    num_newaxis = data_ndim - axis - 1

    # always add in 1-th axis ?? (TODO:wjq)
    #if num_newaxis == 0:
    output_data = lambda on, os: tvm.sum(
        input[on, os].astype(out_dtype) + bias[os].astype(out_dtype),
        axis=[])
    #else:
    #    output_data = lambda on, os: tvm.sum(
    #        input[n, os].astype(out_dtype) + bias[oc])

    return tvm.compute(
            (batch, species),
            output_data, tag="bias_add2d")
            
def bias_add4d_compute(input, bias, axis=1, out_dtype=None):
    if out_dtype is None:
            out_dtype = input.dtype
            
    batch, in_channel, in_height, in_width = input.shape
            
    data_ndim = len(input.shape)
    num_newaxis = data_ndim - axis - 1

    # always add in 1-th axis ?? (TODO:wjq)
    # when axis == 1 : layout = NCHW
    # when axis == 3 : layout == NHWC
    #if num_newaxis == 0:
    output_data = lambda on, oc, oh, ow: tvm.sum(
        input[on, oc, oh, ow].astype(out_dtype) + bias[oc].astype(out_dtype),
        axis=[])
    #else:
    #    output_data = lambda on, os: tvm.sum(
    #        input[n, os].astype(out_dtype) + bias[oc])
    
    return tvm.compute(
            (batch, in_channel, in_height, in_width),
            output_data, tag="bias_add4d")


