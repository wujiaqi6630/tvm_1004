# wjq 2020/09/20
# Schedule and compute definition for dropout operators

import numpy as np
import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from ..nn.dropout import dropout

from ..util import simplify

from tvm import DPU_path as Dp


@autotvm.register_topi_compute(dropout, 'cpu', ['direct'])
def _declaration_dropout (cfg, input, out_dtype):
    #if layout == 'NCHW':
    #return dropout_compute (input, out_dtype)
    data_ndim = len(input.shape)
    if out_dtype is None:
        out_dtype = input.dtype

    if data_ndim == 2:
        x = input.shape[0].value
        y = input.shape[1].value
        data = tvm.placeholder((x,y), name='data', dtype = 'float32')
        random_data = np.random.random((x,y)).astype("float32")
        # copy random data to file
        np.savetxt(Dp.randomArrayPath, random_data)
        
        batch, species = input.shape
        output_data = lambda on, os: tvm.max(
            tvm.expr.Select(
                tvm.all(data[on,os] > 0.5),
                input[on, os].astype(out_dtype),
                0.0),
            #(input[on, os].astype(out_dtype), relay.const(0.0)),
            axis=[])

        return tvm.compute((batch, species), output_data, tag="dropout2d")
        #return dropout2d_compute(input, data, out_dtype)
    elif data_ndim == 4:
        i = input.shape[0].value
        j = input.shape[1].value
        k = input.shape[2].value
        t = input.shape[3].value
        data = tvm.placeholder((i,j,k,t), name='data', dtype = 'float32')
        random_data = np.random.random((i,j,k,t)).astype("float32")
        np.savetxt(Dp.randomArrayPath, random_data)
        return dropout4d_compute(input, data,  out_dtype)
    raise ValueError("Dropout op's input data shape should be 2 or 4.")

def dropout2d_compute (input, data, out_dtype=None):
    if out_dtype is None:
        out_dtype = input.dtype
    
    batch, species = input.shape
    
    output_data = lambda on, os: tvm.max(
            tvm.expr.Select(
                tvm.all(data[on,os] > 0.5),
                input[on, os].astype(out_dtype),
                0.0),
            #(input[on, os].astype(out_dtype), relay.const(0.0)),
            axis=[])

    return tvm.compute((batch, species), output_data, tag="dropout2d")


def dropout4d_compute (input, data, out_dtype=None):
    if out_dtype is None:
        out_dtype = input.dtype
    
    batch, in_channel, in_height, in_width = input.shape
    
    output_data = lambda on, oc, oh, ow: tvm.max(
            tvm.expr.Select(
                tvm.all(data[on, oc, oh, ow] > 0.5),
                input[on, oc, oh, ow].astype(out_dtype),
                0.0),
            #(input[on, oc, oh, ow].astype(out_dtype), relay.const(0.0)),
            axis=[])
    
    return tvm.compute(
        (batch, in_channel, in_height, in_width),
        output_data, tag="dropout4d")

