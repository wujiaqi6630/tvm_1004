# wjq 2020/09/21
# Compute definition for softmax operators

from __future__ import absolute_import
from .. import generic, tag

import tvm

#@tvm.tag_scope(tag='softmax_output')
@tvm.target.generic_func
def softmax(x, axis=-1):
    
    return softmax(x,  axis)
   
@tvm.target.generic_func
def softmaxMax(x, axis=-1):
    return softmaxMax(x,  axis)

@tvm.target.generic_func
def softmaxSum(max_data, in_data,  axis=-1):
    return softmaxSum(max_data, in_data,  axis)

@tvm.target.generic_func
def softmaxDiv(sum_data, max_data, in_data, axis=-1):
    return softmaxDiv(sum_data, max_data, in_data,  axis)

@tvm.tag_scope(tag='log_softmax_output')
def log_softmax(x):
    """Perform log softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        2-D input data

    Returns
    -------
    output : tvm.Tensor
        2-D output with same shape
    """

    assert len(x.shape) == 2, "only support 2-dim log softmax"
    m, n = x.shape
    k = tvm.reduce_axis((0, n), name='k')
    max_elem = tvm.compute((m, ), lambda i: tvm.max(x[i, k], axis=k))
    k = tvm.reduce_axis((0, n), name='k')
    expsum = tvm.compute(
        (m, ), lambda i: tvm.sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return tvm.compute(
        x.shape, lambda i, j: x[i, j] - max_elem[i] - tvm.log(expsum[i]))
