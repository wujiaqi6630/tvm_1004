# wjq 2020/09/16
# Caculation description and schedule for bias_add ops

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag

from ..nn.bias_add import bias_add

from ..util import simplify


@autotvm.register_topi_compute(bias_add, 'dpu', ['direct'])
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
    output_data = lambda on, ok: tvm.sum(
        input[on, ok].astype(out_dtype) + bias[ok].astype(out_dtype),
        axis=[])
    #else:
    #    output_data = lambda on, os: tvm.sum(
    #        input[n, os].astype(out_dtype) + bias[oc])

    return tvm.compute(
            (batch, species),
            output_data, tag="bias_add2D")
            
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
            output_data, tag="bias_add4D")

def DPULoopSplitePragma2D(spliteList):
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

def DPULoopSplitePragma4D(spliteList):
    # [1, [[2,3,-1,4,-4,1]], [1,2]]
    iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow"};
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



@autotvm.register_topi_schedule(generic.schedule_bias_add, 'dpu', ['direct'])
def schedule_bias_add(cfg, outs):
    #Default schedule for 
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

        if 'bias_add2D' in op.tag:
            if cfg:
                if len(op.axis) == 2:
                    on,ok = op.axis
                else:
                    raise ValueError("DPUError: Bias_add op should have 2 axis.")
                assert len(op.reduce_axis) == 0
                
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
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma2D(k))
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

        if 'bias_add4D' in op.tag:
            if cfg:
                if len(op.axis) == 4:
                    on,oc,oh,ow = op.axis
                else:
                    raise ValueError("DPUError: Bias_add op should have 4 axis.")
                assert len(op.reduce_axis) == 0
                
                loopIter = {1:on, 2:oc, 3:oh, 4:ow}
                iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow"}
                
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
                                    s[op].pragma(loopIter[k[0]], DPULoopSplitePragma4D(k))
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

