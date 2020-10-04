import tvm

@tvm.target.generic_func
def bias_add (input, bias, axis=1, out_dtype=None):
    return bias_add(input, bias, axis, out_dtype)