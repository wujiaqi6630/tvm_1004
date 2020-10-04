# wjq 2020/09/20

import tvm

@tvm.target.generic_func
def dropout(input, out_dtype=None):
    #if layout == 'NCHW':
    return dropout(input, out_dtype)
    #raise ValueError("not support this layout {} yet".format(layout))

