# wjq 2020/10/04
# Caculation description and schedule for dense ops
# Dense schedule on dpu

from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas
from ..nn.dense import dense

#from .util import get_fp32_len
from .. import generic, tag, nn
from ..util import traverse_inline, get_const_tuple

@autotvm.register_topi_compute(dense, "dpu", "direct")
def _declaration_dense(cfg, data, weight, bias=None, out_dtype=None):
    
    return dense_pack(cfg, data, weight, bias, out_dtype)


# Declare dense compute with packing weight into cache-friendly layout
#@autotvm.register_topi_compute(dense, "dpu", "direct_pack")
def dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape) # batch, in_dim
    N, _ = get_const_tuple(weight.shape) # out_dim
    
    ok = tvm.reduce_axis((0, K), name="ok")
    C = tvm.compute((M, N),
                    lambda oy, ox: tvm.sum(
                        data[oy, ok].astype(out_dtype) *
                        weight[ok, ox].astype(out_dtype),
                        axis=ok),
                    tag="dense_pack")
    if bias is not None:
        C = tvm.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                        tag=tag.BROADCAST)
    return C


# Declare dense compute without packing weight
#@autotvm.register_topi_compute(dense, "dpu", "direct_nopack")
'''
def dense(cfg, data, weight, bias=None, out_dtype=None):
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    
    k = tvm.reduce_axis((0, K), "k")
    CC = tvm.compute((M, N),
                     lambda i, j: tvm.sum(
                         data[i, k].astype(out_dtype) *
                         weight[j, k].astype(out_dtype), axis=k))

    #kk = tvm.reduce_axis((0, vec), "kk")
    #C = tvm.compute((M, N),
    #                lambda y, x: tvm.sum(CC[y, x, kk], axis=kk),
    #                tag="dense_nopack")
    if bias is not None:
        C = tvm.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                        tag=tag.BROADCAST)

    return C
'''
def DPULoopSplitePragma(spliteList):
    # [1, [[2,3,-1,4,-4,1]], [1,2]]
    iterStr = {1:r"oy", 2:r"ox", 3:r"ok"};
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
    iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow", 5:r"ic", 6:r"kh", 7:r"kw"};
    threadBindStr = {-1:r"blockIdx.z", -2:r"blockIdx.y", -3:r"blockIdx.x",
                      -4:r"threadIdx.z",-5:r"threadIdx.y",-6:r"threadIdx.x"}
    str1 = ""
    if spliteList[1] == 2:
        str1 = str1 + r"loop_split(" + iterStr[spliteList[0]] + r"," + str(spliteList[1]) + r","
        str1 = str1 + str(spliteList[2]) + r":"
        str1 = str1 + threadBindStr[spliteList[3]] + r"," + str(spliteList[4]) + r":local)"
    return str1

@autotvm.register_topi_schedule(generic.schedule_dense, 'dpu', ['direct'])
def schedule_dense(cfg, outs):
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

        if 'dense_pack' in op.tag:
            if cfg:
                if len(op.axis) == 2:
                    oy,ox = op.axis
                else:
                    raise ValueError("DPUError: Dense op should have 2 axis.")
                if len(op.reduce_axis) == 1:
                    ok = op.reduce_axis[0]
                else:
                    raise ValueError("DPUError: Dense op should have 1 reduce_axis.")
                
                loopIter = {1:oy, 2:ox, 3:ok}
                iterStr = {1:r"oy", 2:r"ox", 3:r"ok"};
                
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 3:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 3:
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
                                # TODO:wjq
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

@autotvm.register_topi_schedule(generic.schedule_dense, 'dpu', ['direct_pack'])
def schedule_dense(cfg, outs):
    #Default schedule for 
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    return s

@autotvm.register_topi_schedule(generic.schedule_dense, 'dpu', ['direct_nopack'])
def schedule_dense(cfg, outs):
    #Default schedule for 
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    return s

