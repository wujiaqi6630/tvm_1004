# wjq 2020/09/16
# Schedule and compute definition for relu operators

import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from ..nn.relu import relu

from ..util import simplify

@autotvm.register_topi_compute(relu, 'cpu', ['direct'])
def _declaration_relu (cfg, input, out_dtype):
    #if layout == 'NCHW':
    #return relu_compute (input, out_dtype)
    data_ndim = len(input.shape)

    if data_ndim == 2:
        return relu2d_compute(input, out_dtype)
    elif data_ndim == 4:
        return relu4d_compute(input,  out_dtype)
    raise ValueError("Relu op's input data shape should be 2 or 4.")

def relu2d_compute (input, out_dtype=None):
    if out_dtype is None:
        out_dtype = input.dtype
    
    batch, species = input.shape
    
    output_data = lambda on, os: tvm.max(
            tvm.expr.Select(
                tvm.all(input[on, os] > 0),
                input[on, os].astype(out_dtype),
                0.0),
            #(input[on, os].astype(out_dtype), relay.const(0.0)),
            axis=[])

    return tvm.compute((batch, species), output_data, tag="relu2d")


def relu4d_compute (input, out_dtype=None):
    if out_dtype is None:
        out_dtype = input.dtype
    
    batch, in_channel, in_height, in_width = input.shape
    
    output_data = lambda on, oc, oh, ow: tvm.max(
            tvm.expr.Select(
                tvm.all(input[on, oc, oh, ow] > 0),
                input[on, oc, oh, ow].astype(out_dtype),
                0.0),
            #(input[on, oc, oh, ow].astype(out_dtype), relay.const(0.0)),
            axis=[])
    
    return tvm.compute(
        (batch, in_channel, in_height, in_width),
        output_data, tag="relu4d")

