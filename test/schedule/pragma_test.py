import tvm
import os

input(os.getpid())

n = 16
m = 16
A = tvm.placeholder((n, m), name='A')
k = tvm.reduce_axis((0, n), name='k')
l = tvm.reduce_axis((0, m), name = 'l')

B = tvm.compute((n,), lambda i: tvm.sum(A[i, l], axis=l), name='B')

s = tvm.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=4)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].pragma(ki, "#pragma")

print(tvm.lower(s, [A, B], simple_mode=True))
