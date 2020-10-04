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

@autotvm.register_topi_compute(lrn_sqr, 'dpu', ['direct'])
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

    ok = tvm.reduce_axis((0, size), name='ok')

    sqr_out = lambda on, oc, oh, ow: tvm.sum(
            tvm.expr.Select(
                tvm.all(oc >= radius, oc < (in_channel+radius)),
                data[on, oc-radius+ok, oh, ow].astype(out_dtype) * data[on, oc-radius+ok, oh, ow].astype(out_dtype),
                0.0) ,
            axis=[ok])

    return tvm.compute((batch, in_channel, in_height, in_width), sqr_out, tag="lrn_sqrt")

# TODO : wjq
def lrn_sqr_nhwc(data, size, axis):
    print('TODO: wjq')


@autotvm.register_topi_compute(lrn_pow, 'dpu', ['direct'])
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
    return tvm.compute((batch, in_channel, in_height, in_width), pow_out, tag="lrn_pow")

# TODO : wjq
def lrn_pow_nhwc (data, size, axis, alpha, beta, bias):
    print('TODO: wjq')


@autotvm.register_topi_compute(lrn_div, 'dpu', ['direct'])
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
    return tvm.compute((batch, in_channel, in_height, in_width), div_out, tag="lrn_div")

# TODO : wjq
def lrn_div_nhwc (in_data, pow_data):
    print('TODO: wjq')



@autotvm.register_topi_compute(lrn, 'dpu', ['direct'])
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
    
    # pad and sqrt op:
    output_data1 = lambda on, oc, oh, ow: tvm.sum(
            tvm.expr.Select(
                tvm.all(oc >= radius, oc < (in_channel+radius)),
                data[on, oc-radius+ls, oh, ow].astype(out_dtype) * data[on, oc-radius+ls, oh, ow].astype(out_dtype),
                0.0) ,
            axis=[ls])

    sqr_out = tvm.compute((batch, in_channel, in_height, in_width), output_data1, tag="lrn_sqrt_op")

    #return sqr_out
    
    # pow op:
    output_data2 = lambda on, oc, oh, ow: tvm.power(
            (1 + (alpha / size * sqr_out[on, oc, oh, ow].astype(out_dtype))),
            beta)
    pow_out = tvm.compute((batch, in_channel, in_height, in_width), output_data2, tag="lrn_pow_op")

    #return pow_op
    
    # div op:
    output_data3 = lambda on, oc, oh, ow: tvm.expr.Div(
            data[on, oc, oh, ow].astype(out_dtype),
            pow_out[on, oc, oh, ow].astype(out_dtype))
    div_out = tvm.compute((batch, in_channel, in_height, in_width), output_data3, tag="lrn_div_op")

    return div_out
    #'''
# TODO : wjq
def lrn_nhwc (data, size, axis, alpha, beta, bias):
    print('TODO: wjq')

def DPULoopSplitePragma(spliteList):
    # [1, [[2,3,-1,4,-4,1]], [1,2]]
    iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow", 5:r"ok"};
    threadBindStr = {-1:r"blockIdx.z", -2:r"blockIdx.y", -3:r"blockIdx.x",
                      -4:r"threadIdx.z",-5:r"threadIdx.y",-6:r"threadIdx.x"}
    str1 = ""
    if spliteList[1] == 3:
        str1 = str1 + r"loop_split(" + iterStr[spliteList[0]] + r"," + str(spliteList[1]) + r","
        str1 = str1 + r"*:" + threadBindStr[spliteList[2]] + r"," + str(spliteList[3]) + r":"
        str1 = str1 + threadBindStr[spliteList[4]] + r"," + str(spliteList[5]) + r":local)"
    # (TODO)
    #if spliteList[1] == 2: 
    
    return str1


@autotvm.register_topi_schedule(generic.schedule_lrn_sqr, 'dpu', ['direct'])
def schedule_lrn_sqr(cfg, outs):
    #
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []     
    
    def traverse(op):
        if 'lrn_sqr' in op.tag:
            if cfg:
                if len(op.axis) == 4:
                    on,oc,oh,ow = op.axis
                else:
                    raise ValueError("DPUError: softmax op should have 4 axis.")
                if len(op.reduce_axis) == 1:
                    ok = op.reduce_axis[0]
                else:
                    raise ValueError("DPUError: softmax op should have 1 reduce_axis.")
                
                loopIter = {1:on, 2:oc, 3:oh, 4:ow, 5:ok}
                iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow", 5:r"ok"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 4:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 5:
                                str1 = str1 + loopIter[k]
                            elif isinstance(k, str):
                                str1 = str1 + r"," + k
                            else:
                                raise ValueError("DPUError: Illegal data type.")
                        str1 = str1 + ")"
                        s[op].pragma(loopIter[cfg[1][0]], str1)

                if isinstance(cfg[2], list):
                    if len(cfg[2]) > 0:
                        for k in cfg[2]:
                            if isinstance(k, list):
                                segments = k[1]
                                if segments == 3 and len(k) == 6:
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma(k))
                                # TODO:wjq
                                #if segments == 2:
                else:
                    raise ValueError("DPUError: The second index of cfg should be a list.")

                if isinstance(cfg[3], list):
                    if len(cfg[3]) > 0:
                        for k in cfg[3]:
                            if isinstance(k, int) and k >= 1 and k <= 5:
                                s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

@autotvm.register_topi_schedule(generic.schedule_lrn_pow, 'dpu', ['direct'])
def schedule_lrn_pow(cfg, outs):
    #Default schedule for.
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []     
    
    def traverse(op):
        if 'lrn_pow' in op.tag:
            if cfg:
                if len(op.axis) == 4:
                    on,oc,oh,ow = op.axis
                else:
                    raise ValueError("DPUError: softmax op should have 4 axis.")
                assert len(op.reduce_axis) == 0
                
                loopIter = {1:on, 2:oc, 3:oh, 4:ow}
                iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 4:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 4:
                                str1 = str1 + loopIter[k]
                            elif isinstance(k, str):
                                str1 = str1 + r"," + k
                            else:
                                raise ValueError("DPUError: Illegal data type.")
                        str1 = str1 + ")"
                        s[op].pragma(loopIter[cfg[1][0]], str1)

                if isinstance(cfg[2], list):
                    if len(cfg[2]) > 0:
                        for k in cfg[2]:
                            if isinstance(k, list):
                                segments = k[1]
                                if segments == 3 and len(k) == 6:
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma(k))
                                # TODO:wjq
                                #if segments == 2:
                else:
                    raise ValueError("DPUError: The second index of cfg should be a list.")

                if isinstance(cfg[3], list):
                    if len(cfg[3]) > 0:
                        for k in cfg[3]:
                            if isinstance(k, int) and k >= 1 and k <= 4:
                                s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

@autotvm.register_topi_schedule(generic.schedule_lrn_div, 'dpu', ['direct'])
def schedule_lrn_div(cfg, outs):
    #Default schedule for 
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []     
    
    def traverse(op):
        if 'lrn_div' in op.tag:
            if cfg:
                if len(op.axis) == 4:
                    on,oc,oh,ow = op.axis
                else:
                    raise ValueError("DPUError: softmax op should have 4 axis.")
                assert len(op.reduce_axis) == 0
                
                loopIter = {1:on, 2:oc, 3:oh, 4:ow}
                iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 4:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 4:
                                str1 = str1 + loopIter[k]
                            elif isinstance(k, str):
                                str1 = str1 + r"," + k
                            else:
                                raise ValueError("DPUError: Illegal data type.")
                        str1 = str1 + ")"
                        s[op].pragma(loopIter[cfg[1][0]], str1)

                if isinstance(cfg[2], list):
                    if len(cfg[2]) > 0:
                        for k in cfg[2]:
                            if isinstance(k, list):
                                segments = k[1]
                                if segments == 3 and len(k) == 6:
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma(k))
                                # TODO:wjq
                                #if segments == 2:
                else:
                    raise ValueError("DPUError: The second index of cfg should be a list.")

                if isinstance(cfg[3], list):
                    if len(cfg[3]) > 0:
                        for k in cfg[3]:
                            if isinstance(k, int) and k >= 1 and k <= 4:
                                s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

