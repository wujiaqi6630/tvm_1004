# wjq 2020/09/15
# Schedule and compute definition for pooling operators
import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config

from .. import nn
from ..nn.pooling import max_pool2d


from .. import generic
from .. import tag

from ..util import simplify


@autotvm.register_topi_compute(max_pool2d, 'cpu', ['direct'])
def _declaration_max_pool2d (cfg, input, pool_size, strides, padding, layout, out_dtype):
    pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (strides, strides)
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    #if layout == 'NCHW':
    return max_pool2d_nchw (input, pool_size, strides, padding, out_dtype)
    # TODO(wjq)
    raise ValueError("not support this layout {} yet".format(layout))

def max_pool2d_nchw (input, pool_size, stride, padding, out_dtype=None):
    if out_dtype is None:
        out_dtype = input.dtype
    assert isinstance(pool_size, int) or len(pool_size) == 2
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(padding, int) or len(padding) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(pool_size, int):
        kernel_h = kernel_w = pool_size
    else:
        kernel_h, kernel_w = pool_size

    batch, in_channel, in_height, in_width = input.shape

    out_channel = in_channel
    # In Caffe, when the pooling operator is not divisible, ceil is adopted,
    # while the convolution operator is floor
    #out_height = math.ceil((in_height+2*pad_h-kernel_h)/stride_h+1)
    #out_width = math.ceil((in_width+2*pad_w-kernel_w)/stride_w+1)
    out_height = simplify((in_height+2*pad_h-kernel_h)//stride_h+1)
    out_width = simplify((in_width+2*pad_w-kernel_w)//stride_w+1)

    kh = tvm.reduce_axis((0, kernel_h), name='kh')
    kw = tvm.reduce_axis((0, kernel_w), name='kw')

    output_data = lambda n, oc, oh, ow: tvm.max(
        tvm.expr.Select(
            tvm.all((oh*stride_h+kh) >= pad_h, (oh*stride_h+kh) < (in_height+pad_h), (ow*stride_w+kw >= pad_w), (ow*stride_w+kw < in_width+pad_w)),
            input[n, oc, oh * stride_h + kh - pad_h, ow * stride_w + kw - pad_w].astype(out_dtype),
            0.0),
        axis=[kh, kw])

    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        output_data, tag="max_pool2d_nchw")

def _parallel_sch(sch, oshape, do_vectorize=False):
    def vectorize(fused_axis, num_parallel_axis, vectorize_limit=64):
        """Internal vectorization utility function."""
        reorder_axis = [fused_axis]
        for i in range(num_parallel_axis, len(sch.op.axis) - 1):
            reorder_axis.append(sch.op.axis[i])
        kw, kh = sch.op.reduce_axis
        fuse_k = sch.fuse(kw, kh)
        c = sch.op.axis[len(sch.op.axis) - 1]
        reorder_axis += [fuse_k, c]
        sch.reorder(*reorder_axis)
        inner_length = oshape[len(oshape) - 1].value
        if inner_length <= vectorize_limit:
            sch.vectorize(c)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, c_i = sch.split(c, split_factor)
                sch.vectorize(c_i)

    if len(sch.op.axis) >= 5:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1], sch.op.axis[2])
        if do_vectorize:
            vectorize(fused, 3)

    elif len(sch.op.axis) >= 3:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1])
        if do_vectorize:
            vectorize(fused, 2)
    else:
        sch.parallel(sch.op.axis[0])
        return
    sch.parallel(fused)


@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs, layout):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, tvm.tensor.ComputeOp):
            s[PaddedInput].compute_inline()
        do_vectorize = layout[-1] not in "HWhw"
        _parallel_sch(s[Pool], outs[0].shape, do_vectorize)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('pool') or OP.tag.startswith('max') or OP.tag.startswith('avg'):
            # Average pool accumulation and division happens in different for loops (#3607).
            # To ensure good parallel support, apply multi-threading on the second loop.
            if OP != outs[0].op:
                output = outs[0]
                output_fused = s[output].fuse(output.op.axis[0], output.op.axis[1])
                s[output].parallel(output_fused)

            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


@generic.schedule_adaptive_pool.register(["cpu"])
def schedule_adaptive_pool(outs):
    """Schedule for adaptive pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('adaptive_pool'):
            if OP != outs[0].op:
                output = outs[0]
                output_fused = s[output].fuse(output.op.axis[0], output.op.axis[1])
                s[output].parallel(output_fused)

            Pool = OP.output(0)
            _parallel_sch(s[Pool], outs[0].shape)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
