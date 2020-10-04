# wjq 2020/09/21
# Schedule and compute definition for softmax operators

import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from ..nn.softmax import softmax, softmaxMax, softmaxSum, softmaxDiv
from .. import generic, tag

from ..util import simplify

from itertools import product
import numpy as np


@autotvm.register_topi_compute(softmax, 'dpu', ['direct'])
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


@autotvm.register_topi_compute(softmaxMax, 'dpu', ['direct'])
def _declaration_softmaxMax(cfg, x, axis):
    
    assert len(x.shape) == 2
    m, n = x.shape
    ok = tvm.reduce_axis((0, n), name='ok')
    maxelem = tvm.compute((m, ), lambda on: tvm.max(x[on, ok], axis=ok), tag="softmaxMax")

    return maxelem

@autotvm.register_topi_compute(softmaxSum, 'dpu', ['direct'])    
def _declaration_softmaxSum(cfg, max_data, in_data, axis):

    assert len(in_data.shape) == 2
    m, n = in_data.shape
    
    ok = tvm.reduce_axis((0, n), name='ok')
    expsum = tvm.compute(
        (m, ), lambda on: tvm.sum(tvm.exp(in_data[on,ok] - max_data[on,0]), axis=ok), tag="softmaxSum")

    return expsum

@autotvm.register_topi_compute(softmaxDiv, 'dpu', ['direct'])
def _declaration_softmaxDiv(cfg, sum_data, max_data, in_data, axis):

    assert len(in_data.shape) == 2
    m, n = in_data.shape
    
    divelem = tvm.compute(
        in_data.shape, lambda on, ok: ((tvm.exp(in_data[on,ok] - max_data[on,0])) / sum_data[on,0]), tag="softmaxDiv")

    return divelem

def DPULoopSplitePragma(spliteList):
    # [1, [[2,3,-1,4,-4,1]], [1,2]]
    iterStr = {1:r"on", 2:r"ok"};
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

def DPUReductionLoopSplitePragma(spliteList):
    # [2,2,1,-1,1024]
    iterStr = {1:r"on", 2:r"ok"};
    threadBindStr = {-1:r"blockIdx.z", -2:r"blockIdx.y", -3:r"blockIdx.x",
                      -4:r"threadIdx.z",-5:r"threadIdx.y",-6:r"threadIdx.x"}
    str1 = ""
    if spliteList[1] == 2:
        str1 = str1 + r"loop_split(" + iterStr[spliteList[0]] + r"," + str(spliteList[1]) + r","
        str1 = str1 + str(spliteList[2]) + r":"
        str1 = str1 + threadBindStr[spliteList[3]] + r"," + str(spliteList[4]) + r":local)"

    return str1


@autotvm.register_topi_schedule(generic.schedule_softmaxMax, 'dpu', ['direct'])
def schedule_softmaxMax(cfg, outs):
    # Default schedule of softmaxMax for DPU
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []     
    
    def traverse(op):        
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # [1, [[2,3,-1,64,-4,16]],[1,"compute","fmax"], [1,2]],
        if 'softmaxMax' in op.tag:
            if cfg:
                if len(op.axis) == 1:
                    on = op.axis[0]
                else:
                    raise ValueError("DPUError: softmax op should have 1 axis.")
                if len(op.reduce_axis) == 1:
                    ok = op.reduce_axis[0]
                else:
                    raise ValueError("DPUError: softmax op should have 1 reduce_axis.")
                
                loopIter = {1:on, 2:ok}
                iterStr = {1:r"on", 2:r"ok"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 2:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 2:
                                str1 = str1 + iterStr[k]
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
                                elif segments == 2 and len(k) == 5:
                                    s[op].pragma(loopIter[k[0]], DPUReductionLoopSplitePragma(k))
                                else:
                                    raise ValueError("DPUError: Other segmentation cases are not supported.")
                else:
                    raise ValueError("DPUError: The second index of cfg should be a list.")

                if isinstance(cfg[3], list):
                    if len(cfg[3]) > 0:
                        for k in cfg[3]:
                            if isinstance(k, int) and k >= 1 and k <= 2:
                                s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

@autotvm.register_topi_schedule(generic.schedule_softmaxSum, 'dpu', ['direct'])
def schedule_softmaxSum(cfg, outs):
    #Default schedule of softmaxSum for DPU
    assert cfg != None
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []     
    
    def traverse(op):
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'softmaxSum' in op.tag:
            if cfg:
                if len(op.axis) == 1:
                    on = op.axis[0]
                else:
                    raise ValueError("DPUError: softmax op should have 1 axis.")
                if len(op.reduce_axis) == 1:
                    ok = op.reduce_axis[0]
                else:
                    raise ValueError("DPUError: softmax op should have 1 reduce_axis.")
                
                loopIter = {1:on, 2:ok}
                iterStr = {1:r"on", 2:r"ok"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 2:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 2:
                                str1 = str1 + iterStr[k]
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
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma(k), tag="DPU")
                                elif segments == 2 and len(k) == 5:
                                    s[op].pragma(loopIter[k[0]], DPUReductionLoopSplitePragma(k), tag="DPU")
                                else:
                                    raise ValueError("DPUError: Other segmentation cases are not supported.")
                else:
                    raise ValueError("DPUError: The second index of cfg should be a list.")

                if isinstance(cfg[3], list):
                    if len(cfg[3]) > 0:
                        for k in cfg[3]:
                            if isinstance(k, int) and k >= 1 and k <= 2:
                                s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

@autotvm.register_topi_schedule(generic.schedule_softmaxDiv, 'dpu', ['direct'])
def schedule_softmaxDiv(cfg, outs):
    #Default schedule for 
    assert cfg != None
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []     
    
    def traverse(op):
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'softmaxDiv' in op.tag:
            if cfg:
                if len(op.axis) == 2:
                    on,ok = op.axis
                else:
                    raise ValueError("DPUError: softmax op should have 2 axis.")
                assert len(op.reduce_axis) == 0
                
                loopIter = {1:on, 2:ok}
                iterStr = {1:r"on", 2:r"ok"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 2:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD", tag="DPU")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 2:
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
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma(k), tag="DPU")
                                # TODO:wjq
                                #if segments == 2:
                else:
                    raise ValueError("DPUError: The second index of cfg should be a list.")

                if isinstance(cfg[3], list):
                    if len(cfg[3]) > 0:
                        for k in cfg[3]:
                            if isinstance(k, int) and k >= 1 and k <= 2:
                                s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

