# wjq 2020/09/21
# Schedule and compute definition for softmax operators

import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from ..nn.softmax import softmax, softmaxMax, softmaxSum, softmaxDiv

from ..util import simplify

from itertools import product
import numpy as np


@autotvm.register_topi_compute(softmax, 'cpu', ['direct'])
def _declaration_softmax (cfg, x, axis):
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        ValueError("axis parameter should be less than input dim")
    if axis == 1: 
        return softmax(x, axis)
    # (TODO : wjq, axis == 0 ?)
    raise ValueError("Softmax op's layout should be in ['NCHW', 'NHWC'].")

def softmax(x, axis):
    
    assert len(x.shape) == 2
    m, n = x.shape
    ok = tvm.reduce_axis((0, n), name='ok')
    maxelem = tvm.compute((m, ), lambda on: tvm.max(x[on, ok], axis=ok))
    ok = tvm.reduce_axis((0, n), name='ok')
    expsum = tvm.compute(
        (m, ), lambda on: tvm.sum(tvm.exp(x[on, ok] - maxelem[on]), axis=ok)) 
    divelem = tvm.compute(
        x.shape, lambda on, os: (tvm.exp(x[on,os] - maxelem[on]) / expsum[on]))
    return divelem


@autotvm.register_topi_compute(softmaxMax, 'cpu', ['direct'])
def _declaration_softmaxMax(cfg, x, axis):
    
    assert len(x.shape) == 2
    m, n = x.shape
    ok = tvm.reduce_axis((0, n), name='ok')
    maxelem = tvm.compute((m, ), lambda on: tvm.max(x[on, ok], axis=ok))

    return maxelem

@autotvm.register_topi_compute(softmaxSum, 'cpu', ['direct'])    
def _declaration_softmaxSum(cfg, max_data, in_data, axis):

    assert len(in_data.shape) == 2
    m, n = in_data.shape
    
    ok = tvm.reduce_axis((0, n), name='ok')
    expsum = tvm.compute(
        (m, ), lambda on: tvm.sum(tvm.exp(in_data[on,ok] - max_data[on,0]), axis=ok))

    return expsum

@autotvm.register_topi_compute(softmaxDiv, 'cpu', ['direct'])
def _declaration_softmaxDiv(cfg, sum_data, max_data, in_data, axis):

    assert len(in_data.shape) == 2
    m, n = in_data.shape
    
    divelem = tvm.compute(
        in_data.shape, lambda on, os: ((tvm.exp(in_data[on,os] - max_data[on,0])) / sum_data[on,0]))

    return divelem
    '''
    assert len(x.shape) == 2
    shape = x.shape
    ok = tvm.reduce_axis((0, shape[axis]), name='ok')

    # max op:
    def insert_reduce_index(indices, reduce_index):
        return indices[:axis] + (reduce_index,) + indices[axis:]

    def _compute_max(*indices):
        eval_range = insert_reduce_index(indices, ok)
        return tvm.max(x[eval_range], axis=ok)
        
    reduced_shape = tuple([dim for (i, dim) in enumerate(shape) if i != axis])
    max_elem = tvm.compute(reduced_shape, _compute_max, tag='softmax_output')

    # sumexp op:
    expsum = lambda on: tvm.sum(
            tvm.exp(x[on,ok]-max_elem[on]),
            axis=[ok])
    sumexp = tvm.compute(reduced_shape, expsum, tag='softmax_output')
    
    # div op:
    _normalize = lambda on, os: tvm.expr.Div(
            tvm.exp(x[on,os]-max_elem[on]),
            sumexp[on])
            
    return tvm.compute(shape, _normalize, tag='softmax_output')
    '''

'''
@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    softmax = outs[0]
    s = tvm.create_schedule([x.op for x in outs])

    return s
'''

