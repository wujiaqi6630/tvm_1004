from __future__ import absolute_import, print_function
import tvm
import numpy as np
import pdb

import os
input(os.getpid())

tgt_host="llvm"
tgt="dpu"



n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

s = tvm.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=31)

if tgt == "dpu":
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))



#pdb.set_trace()

#pdb.set_trace()
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")\


print("-----DPU code-----")

ctx = tvm.context(tgt, 0)
n = 1022
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
#fadd(a, b, c)

if tgt == "dpu":
    dev_module = fadd.imported_modules[0]
    print(dev_module.get_source())
else:
    print(fadd.get_source())