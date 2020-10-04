# wjq 2020/09/17
# Compute definition for lrn operators
import tvm

@tvm.target.generic_func
def lrn(data, size, axis, alpha, beta, bias):
    return lrn(data, size, axis, alpha, beta, bias)
    #raise ValueError("not support this layout {} yet".format(layout))

@tvm.target.generic_func
def lrn_sqr(data, size, axis):
    return lrn_sqr(data, size, axis, alpha, beta, bias)
    #raise ValueError("not support this layout {} yet".format(layout))

@tvm.target.generic_func
def lrn_pow(data, size, axis, alpha, beta, bias):
    return lrn_pow(data, size, axis, alpha, beta, bias)
    #raise ValueError("not support this layout {} yet".format(layout))

@tvm.target.generic_func
def lrn_div(pow_data, in_data, axis):
    return lrn_div(pow_data, in_data, axis)
    #raise ValueError("not support this layout {} yet".format(layout))

