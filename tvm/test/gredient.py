import numpy as np
from sympy import *

def gredientTest(x,w,o,rate):
    #o = conv(x,w,o)
    w_w,w_h = w.shape
    o_w,o_h = o.shape
    for i in range(w_w):
        for j in range(w_h):
            w[i][j] = 0
            y = symbols('y')
            y = np.sum(x[i:i+w_h, j:j+w_w] * w)
            w = w[i][j] + diff(o[k][m],w[i][j])
    return w

def conv(x,w,o):
    weight_w,weight_h = w.shape
    output_w,output_h = o.shape  
    for i in range(output_w):
        for j in range(output_h):
            output[i][j] = np.sum(x[i:i+weight_h, j:j+weight_w] * w)
    return output


if __name__ == "__main__":
    data = np.array([(1,2,3),(2,3,4),(3,4,5)]).astype("float32")
    weight = np.array([(1,2),(2,3)]).astype("float32")
    input_h,input_w = data.shape
    weight_h,weight_w = weight.shape
    padding = 0
    strides = 1
    output_h = (input_h - weight_h + 2*padding) // strides + 1
    output_w = (input_w - weight_w + 2*padding) // strides + 1
    output = np.random.uniform(-1, 1, size=(output_w,output_h)).astype("float32")
    #result = conv(data,weight,output)
    #print(result)
    learning_rate = 0.5
    gredient_result = gredientTest(data,weight,output,learning_rate)
    print(gredient_result)
