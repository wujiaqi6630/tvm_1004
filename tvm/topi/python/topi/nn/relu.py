# wjq 2020/09/16

import tvm

@tvm.target.generic_func
def relu(input, out_dtype=None):
    #if layout == 'NCHW':
    return relu(input, out_dtype)
    #raise ValueError("not support this layout {} yet".format(layout))

