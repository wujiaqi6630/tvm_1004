# wjq 2020/10/04
# Caculation description and schedule for conv2d ops
# Conv2D schedule on dpu

import logging
import re

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..nn.conv2d import conv2d, conv2d_NCHWc, \
    conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.pad import pad
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple

#wjq
from ..util import simplify

logger = logging.getLogger('topi')
#@autotvm.register_topi_compute(nn.conv2d, ['cuda', 'gpu'], ['direct', 'winograd', 'int8'])
#def conv2d_cuda(cfg, data, kernel, strides, padding, dilation, layout='NCHW', out_dtype='float32'):

@autotvm.register_topi_compute(nn.conv2d, 'dpu', ['direct'])
#def conv2d_cuda(cfg, data, kernel, strides, padding, dilation, layout='NCHW', out_dtype='float32'):
def conv2d_dpu(cfg, data, kernel, strides, padding, dilation, layout='NCHW', out_dtype='float32'):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    if layout == 'NCHW':
        return conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))

def conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    #temp = pad(Input, pad_before, pad_after, name="pad_temp")

    #hs = oh * stride_h - (pad_top + pad_down)
    #ws = ow * stride_w - (pad_left + pad_right)
    #hend = min(kernel_h, in_height - hs)
    #wend = min(kernel_w, in_width - ws)
    #hstart = max(-hs, 0)
    #wstart = max(-ws, 0)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_h), name='kh')
    kw = tvm.reduce_axis((0, kernel_w), name='kw')

    output_data = lambda on, oc, oh, ow: tvm.sum(
        tvm.expr.Select(
            tvm.all((oh*stride_h+kh*dilation_h) >= pad_top, (oh*stride_h+kh*dilation_h) < (in_height+pad_top), (ow*stride_w+kw*dilation_w >= pad_left), (ow*stride_w+kw*dilation_w < in_width+pad_left)),
            Input[on, ic, oh * stride_h + kh * dilation_h - pad_top, ow * stride_w + kw * dilation_w - pad_left].astype(out_dtype) * Filter[oc, ic, kh, kw].astype(out_dtype),
            0.0 * Filter[oc, ic, kh, kw].astype(out_dtype)),
        axis=[ic, kh, kw])

    """
    output_data = lambda bh, oc, oh, ow: tvm.sum(
            temp[bh, dic, oh * stride_h + dkh * dilation_h,
                 ow * stride_w + dkw * dilation_w].astype(out_dtype) *
            Filter[oc, dic, dkh, dkw].astype(out_dtype),
            axis=[dic, dkh, dkw])
    """
    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        output_data, tag="conv2d_nchw")
    """
    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: tvm.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw")
    """

@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, 'dpu', ['direct'])
def schedule_conv2d(cfg, outs):

    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def DPULoopSplitePragma(spliteList):
        iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow", 5:r"ic", 6:r"kh", 7:r"kw"};
        threadBindStr = {-1:r"blockIdx.z", -2:r"blockIdx.y", -3:r"blockIdx.x",
                          -4:r"threadIdx.z",-5:r"threadIdx.y",-6:r"threadIdx.x"}
        str1 = ""
        if spliteList[1] == 3:
            str1 = str1 + r"loop_split(" + iterStr[spliteList[0]] + r"," + str(spliteList[1]) + r","
            str1 = str1 + r"*:" + threadBindStr[spliteList[2]] + r"," + str(spliteList[3]) + r":"
            str1 = str1 + threadBindStr[spliteList[4]] + r"," + str(spliteList[5]) + r":local)"
        
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
    
    def traverse(op):
        #Traverse operators from computation graph
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_nchw' in op.tag:
            if cfg:
                on, oc, oh, ow = op.axis
                ic, kh, kw = op.reduce_axis
                
                loopIter = {1:on, 2:oc, 3:oh, 4:ow, 5:ic, 6:kh, 7:kw}
                iterStr = {1:r"on", 2:r"oc", 3:r"oh", 4:r"ow", 5:r"ic", 6:r"kh", 7:r"kw"}
                if len(cfg) != 4:
                    raise ValueError("DPUError: The length of cfg should be 4.")

                if isinstance(cfg[0], int) and cfg[0] >= 0 and cfg[0] <= 7:
                    if cfg[0] != 0:
                        s[op].pragma(loopIter[cfg[0]], "SIMD")
                else:
                    raise ValueError("DPUError: Exceeding the maximum nums of loop layers.")

                if isinstance(cfg[1], list):
                    if len(cfg[1]) > 0:
                        str1 = r"reduction("
                        for k in cfg[1]:
                            if isinstance(k, int) and k >= 1 and k <= 7:
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
                    for k in cfg[3]:
                        if isinstance(k, int) and k >= 1 and k <= 7:
                            s[op].pragma(loopIter[k], "unroll", tag="DPU")
                else:
                    raise ValueError("DPUError: The thrid index of cfg should be a list.")

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

