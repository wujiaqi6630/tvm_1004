import pdb
import tvm
import numpy as np
import timeit
import os
from tvm.contrib import util

input(os.getpid())
tgt_host="cpu"
#device = "cpu"

M = tvm.var("n")
K = tvm.var("n")
N = tvm.var("n")

A = tvm.placeholder((M, K), name='A',dtype='float32')
B = tvm.placeholder((K, N), name='B',dtype='float32')

k = tvm.reduce_axis((0, K), 'k')
C = tvm.compute((M, N), lambda i, j: 0, name='C')

s = tvm.create_schedule(C.op)
func = tvm.build(s, [A, B, C], "c",name='DPUGemm')
print(tvm.lower(s, [A, B, C], simple_mode=True))

#print(func.get_source())
batch = 2
in_channels = 3
out_channels = 2
in_height = 6
in_width = 6
out_height = 3
out_width = 3
kernel_height = 2;
kernel_width = 2

A = tvm.placeholder((batch, in_channels, in_height, in_width), name='A',dtype='float32')
B = tvm.placeholder((out_channels, in_channels, kernel_height, kernel_width), name='B',dtype='float32')
C = tvm.compute((1,), tvm.sum(
            A[bh, ic, oh * stride_h + kh * dilation_h,
                 ow * stride_w + kw * dilation_w].astype(out_dtype) *
            B[oc, ic, kh, kw].astype(out_dtype),
            axis=[ic, kh, kw]))

#input = tvm.placeholder((batch,out_channels,out_height,out_width), name='input',dtype='float32')
#output = tvm.compute((batch, out_channels, out_height, out_width),lambda bh, oc, oh, ow: temp, tag="conv2d_nchw")

s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A,B,C], simple_mode=True))



"""
ic = tvm.reduce_axis((0, in_channel), name='ic')
kh = tvm.reduce_axis((0, kernel_h), name='kh')
kw = tvm.reduce_axis((0, kernel_w), name='kw')
    
output_data = lambda bh, oc, oh, ow: tvm.sum(
        temp[bh, ic, oh * stride_h + kh * dilation_h,
             ow * stride_w + kw * dilation_w].astype(out_dtype) *
        Filter[oc, ic, kh, kw].astype(out_dtype),
        axis=[ic, kh, kw])
return tvm.compute(
        (batch, out_channel, out_height, out_width),
        output_data, tag="conv2d_nchw")
"""
# declare some variables for use later
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((m,), name='A')
B = tvm.placeholder((n,), name='B')

C = tvm.compute((m,), lambda i: A[i]+1, name='B')
D = tvm.compute((n,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print(tvm.lower(s, [A, B, C], simple_mode=True))


A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+1, name='B')
C = tvm.compute((m,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+1, name='B')
C = tvm.compute((m,), lambda i: B[i]*2, name='C')

s = tvm.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))
func = tvm.build(s, [A, B, C], "c",name='DPUGemm')
print(func.get_source())




