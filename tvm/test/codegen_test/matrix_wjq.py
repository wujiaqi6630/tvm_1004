import pdb
import tvm
import numpy as np
import timeit
import os

input(os.getpid())
tgt_host="c"
device = "dpu"

M = tvm.var("M")
K = tvm.var("K")
N = tvm.var("N")

A = tvm.placeholder((M, K), name='A',dtype='float32')
B = tvm.placeholder((K, N), name='B',dtype='float32')

k = tvm.reduce_axis((0, K), 'k')
C = tvm.compute((M, N), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')

s = tvm.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target_host=tgt_host,name='DPUGemm')

#print("------------------DPU_LOWER code---------------------")
#print(tvm.lower(s, [A, B, C], simple_mode=True))

"""
scale = 4
num_thread = 8
block_factor = scale * num_thread

block_x = tvm.thread_axis("blockIdx.x")
thread_x = tvm.thread_axis( "threadIdx.x")
block_y = tvm.thread_axis("blockIdx.y")
thread_y = tvm.thread_axis( "threadIdx.y")

by, yi = s[C].split(C.op.axis[0], factor=block_factor)
bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
s[C].bind(by, block_y)
s[C].bind(bx, block_x)
ty, py = s[C].split(yi, factor=scale)
tx, px = s[C].split(xi, factor=scale)
s[C].bind(ty, thread_y)
s[C].bind(tx, thread_x)
#"""

func = tvm.build(s, [A, B, C], target_host=tgt_host,name='DPUGemm')

#"""
#ctx = tvm.context(tgt_host, 0)
M = 32
K = 30
N = 34
a = tvm.nd.array(np.random.uniform(1, 10, size=(M, K)).astype(A.dtype))
b = tvm.nd.array(np.random.uniform(1, 10, size=(K, N)).astype(B.dtype))
c = tvm.nd.array(np.zeros((M,N), dtype=C.dtype))
#func(a,b,c)
#"""
#print("---------------------KERNEL_CODE------------------------")
#dev_module = func.imported_modules[0]
#print(dev_module.get_source())
print("---------------------HOST_CODE------------------------")
print(func.get_source())

