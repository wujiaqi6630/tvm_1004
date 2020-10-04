# wjq 2020/09/17
# Schedule and compute definition for lrn operators

import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from ..nn.lrn import lrn, lrn_sqr, lrn_pow, lrn_div

from ..util import simplify
from .. import generic


from itertools import product
import numpy as np

@autotvm.register_topi_compute(lrn_sqr, 'cpu', ['direct'])
def _declaration_lrn_sqr (cfg, data, size, axis):
    #NCHW layout:
    if axis == 1: 
        return lrn_sqr_nchw(data, size, axis)
    #NHWC layout:
    # (TODO : wjq, other layout ?)
    elif axis == 3:
        return lrn_sqr_nhwc(data, size, axis)
    raise ValueError("LRN op's layout should be in ['NCHW', 'NHWC'].")

def lrn_sqr_nchw(data, size, axis):
    out_dtype = data.dtype
    radius = size // 2
    batch, in_channel, in_height, in_width = data.shape

    ls = tvm.reduce_axis((0, size), name='ls')

    sqr_out = lambda on, oc, oh, ow: tvm.sum(
            tvm.expr.Select(
                tvm.all(oc >= radius, oc < (in_channel+radius)),
                data[on, oc-radius+ls, oh, ow].astype(out_dtype) * data[on, oc-radius+ls, oh, ow].astype(out_dtype),
                0.0) ,
            axis=[ls])

    return tvm.compute((batch, in_channel, in_height, in_width), sqr_out, tag="lrn_sqrt_op")

# TODO : wjq
def lrn_sqr_nhwc(data, size, axis):
    print('TODO: wjq')


@autotvm.register_topi_compute(lrn_pow, 'cpu', ['direct'])
def _declaration_lrn_pow (cfg, data, size, axis, alpha, beta, bias):
    #NCHW layout:
    if axis == 1: 
        return lrn_pow_nchw(data, size, axis, alpha, beta, bias)
    #NHWC layout:
    # (TODO : wjq, other layout ?)
    elif axis == 3:
        return lrn_pow_nhwc(data, size, axis, alpha, beta, bias)
    raise ValueError("LRN op's layout should be in ['NCHW', 'NHWC'].")

def lrn_pow_nchw(data, size, axis, alpha, beta, bias):
    out_dtype = data.dtype
    batch, in_channel, in_height, in_width = data.shape
    
    pow_out = lambda on, oc, oh, ow: tvm.power(
            (1 + (alpha / size * data[on, oc, oh, ow].astype(out_dtype))),
            beta)
    return tvm.compute((batch, in_channel, in_height, in_width), pow_out, tag="lrn_pow_op")

# TODO : wjq
def lrn_pow_nhwc (data, size, axis, alpha, beta, bias):
    print('TODO: wjq')


@autotvm.register_topi_compute(lrn_div, 'cpu', ['direct'])
def _declaration_lrn_div (cfg, pow_data, in_data, axis):
    #NCHW layout:
    if axis == 1: 
        return lrn_div_nchw(in_data, pow_data)
    #NHWC layout:
    # (TODO : wjq, other layout ?)
    elif axis == 3:
        return lrn_div_nhwc(in_data, pow_data)
    raise ValueError("LRN op's layout should be in ['NCHW', 'NHWC'].")

def lrn_div_nchw(in_data, pow_data):

    out_dtype = in_data.dtype
    batch, in_channel, in_height, in_width = in_data.shape
    
    div_out = lambda on, oc, oh, ow: tvm.expr.Div(
            in_data[on, oc, oh, ow].astype(out_dtype),
            pow_data[on, oc, oh, ow].astype(out_dtype))
    return tvm.compute((batch, in_channel, in_height, in_width), div_out, tag="lrn_div_op")

# TODO : wjq
def lrn_div_nhwc (in_data, pow_data):
    print('TODO: wjq')



@autotvm.register_topi_compute(lrn, 'cpu', ['direct'])
def _declaration_lrn (cfg, data, size, axis, alpha, beta, bias):
    #radius = size // 2
    #NCHW layout:
    if axis == 1: 
        return lrn_nchw(data, size, axis, alpha, beta, bias)
    #NHWC layout:
    # (TODO : wjq, other layout ?)
    elif axis == 3:
        return lrn_nhwc(data, size, axis, bias, alpha, beta)
    raise ValueError("LRN op's layout should be in ['NCHW', 'NHWC'].")

def lrn_nchw (data, size, axis, alpha, beta, bias):
    #default : size = 5, axis=1, alpha=0.0001, beta=0.75, bias=1
    #sqrt_out = lrn_sqrt()
    #pow_out = lrn_pow()
    #div_out = lrn_div()
    #return div_out
    out_dtype = data.dtype
    radius = size // 2
    batch, in_channel, in_height, in_width = data.shape
    ls = tvm.reduce_axis((0, size), name='ls')
    '''
    output_data1 = lambda on, oc, oh, ow: tvm.sum(
                tvm.expr.Select(
                    tvm.all(oc >= radius, oc < (in_channel+radius)),
                    data[on, oc-radius+ls, oh, ow].astype(out_dtype) * A[on],
                    0.0) ,
                axis=[ls])

    sqrt_out = tvm.compute((batch, in_channel, in_height, in_width), output_data1, tag="lrn_sqrt_op")
   
    return sqrt_out
    '''

    '''
    # fuse 'pad , sqrt , pow , div op' into one loop
    
    output_data = lambda on, oc, oh, ow: tvm.expr.Div(
        data[on, oc, oh, ow].astype(out_dtype),
        (tvm.expr.Pow(
                (1 + (alpha / size * (tvm.sum(
                                            tvm.expr.Select(
                                                tvm.all(oc >= radius, oc < (in_channel+radius)),
                                                data[on, oc-radius+ls, oh, ow].astype(out_dtype) * data[on, oc-radius+ls, oh, ow].astype(out_dtype),
                                                0.0) ,
                                            axis=[ls])
                                      )
                    )),beta
                )
        )
    )
    return tvm.compute((batch, in_channel, in_height, in_width), output_data, tag="lrn_compute")
    '''
    # pad and sqrt op:
    #def lrn_fuse_pad_sqrt_op():
    output_data1 = lambda on, oc, oh, ow: tvm.sum(
            tvm.expr.Select(
                tvm.all(oc >= radius, oc < (in_channel+radius)),
                data[on, oc-radius+ls, oh, ow].astype(out_dtype) * data[on, oc-radius+ls, oh, ow].astype(out_dtype),
                0.0) ,
            axis=[ls])

    sqr_out = tvm.compute((batch, in_channel, in_height, in_width), output_data1, tag="lrn_sqrt_op")

    #return sqr_out
    
    # pow op:
    #def lrn_pow_op():
    output_data2 = lambda on, oc, oh, ow: tvm.power(
            (1 + (alpha / size * sqr_out[on, oc, oh, ow].astype(out_dtype))),
            beta)
    pow_out = tvm.compute((batch, in_channel, in_height, in_width), output_data2, tag="lrn_pow_op")

    #return pow_op
    
    # div op:
    #def lrn_div_op():
    output_data3 = lambda on, oc, oh, ow: tvm.expr.Div(
            data[on, oc, oh, ow].astype(out_dtype),
            pow_out[on, oc, oh, ow].astype(out_dtype))
    div_out = tvm.compute((batch, in_channel, in_height, in_width), output_data3, tag="lrn_div_op")

    return div_out
    #'''
# TODO : wjq
def lrn_nhwc (data, size, axis, alpha, beta, bias):
    print('TODO: wjq')


@autotvm.register_topi_schedule(generic.schedule_lrn_sqr, 'cpu', ['direct'])
def schedule_lrn_sqr(cfg, outs):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    return s

@autotvm.register_topi_schedule(generic.schedule_lrn_pow, 'cpu', ['direct'])
def schedule_lrn_pow(cfg, outs):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    return s

@autotvm.register_topi_schedule(generic.schedule_lrn_div, 'cpu', ['direct'])
def schedule_lrn_div(cfg, outs):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    return s

