from numba import njit
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tvm
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
import topi
import os
from time import time
input(os.getpid())

from decimal import *
np.set_printoptions(threshold=np.nan)

def checkTrainingResult(out,predict_out):
    outList = out.tolist()
    count = 0
    for i in range(out.shape[0]):
        if (int(outList[i].index(max(outList[i]))) == int(predict_out[i])):
            count += 1
    return float(count)

def getLenetParams(diamension,str, I, J, K, T):
    if(diamension != 2 and diamension != 4):
        print("Diamension must be 2 or 4 !")
        exit()
    address = "/home/wujiaqi/test/Lenet5/" + str
    f = open(address)
    line = f.readline()
    temp = np.zeros((I*J*K*T))
    i = 0
    while line:
        line1 = line.replace('[','')
        line2 = line1.replace(']','')
        line3 = line2.replace(',','')
        line4 = line3.strip()
        for x in line4.split():
            c = Decimal(x)
            num = float(c)
            temp[i] = num
            i = i+1
        line = f.readline()
    f.close()
    data = np.zeros((I, J, K, T),dtype=np.float32)
    c = 0
    if diamension == 2:
        data = np.zeros((I, J),dtype=np.float32)
        for i in range(I):
            for j in range(J):
                data[i][j] = temp[c]
                c += 1
        return data
    if diamension ==4:
        data = np.zeros((I, J, K, T),dtype=np.float32)
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    for t in range(T):
                        data[i][j][k][t] = temp[c]
                        c += 1
        return data

batch_size = 10000
data = relay.var("data", relay.TensorType((batch_size,1,28,28), "float32"))
data_shape=(batch_size,1,28,28)

conv1_weight = relay.var("conv1_weight")
conv2_weight = relay.var("conv2_weight")
dense1_weight = relay.var("dense1_weight")
dense2_weight = relay.var("dense2_weight")

simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(5,5), channels=20, strides=(1,1),padding=(0, 0))
simple_net = relay.nn.max_pool2d(simple_net,pool_size=(2, 2),strides=(2, 2),padding=(0, 0))
simple_net = relay.nn.conv2d(simple_net, weight=conv2_weight, kernel_size=(5,5), channels=50, strides=(1,1),padding=(0, 0))
simple_net = relay.nn.max_pool2d(simple_net,pool_size=(2, 2),strides=(2, 2),padding=(0, 0))
simple_net = relay.nn.batch_flatten(simple_net)
simple_net = relay.nn.dense(simple_net, dense1_weight,units=500)
simple_net = relay.nn.relu(simple_net)
simple_net = relay.nn.dense(simple_net, dense2_weight,units=10)
simple_net = relay.nn.softmax(simple_net,1)

node = relay.analysis.free_vars(simple_net)
simple_net = relay.Function(node, simple_net)
net, params = testing.create_workload(simple_net)
opt_level = 0
target = tvm.target.cuda()
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(net, target, params=params)
ctx = tvm.gpu()

images = topi.image.load_test_images()
labels = topi.image.load_test_labels()
data = images[0:batch_size]
predict_out = labels[0:batch_size]

paramsName_pair = relay.build_module.getParamsNamePair()

conv1_weight_params = getLenetParams(4,"p0_params.txt",20,1,5,5)
conv2_weight_params = getLenetParams(4,"p1_params.txt",50,20,5,5)
dense1_weight_params = getLenetParams(2,"p2_params.txt",500,800,1,1)
dense2_weight_params = getLenetParams(2,"p3_params.txt",10,500,1,1)

params[paramsName_pair['conv1_weight']] = tvm.nd.array(conv1_weight_params, ctx=tvm.cpu(0))
params[paramsName_pair['conv2_weight']] = tvm.nd.array(conv2_weight_params, ctx=tvm.cpu(0))
params[paramsName_pair['dense1_weight']] = tvm.nd.array(dense1_weight_params, ctx=tvm.cpu(0))
params[paramsName_pair['dense2_weight']] = tvm.nd.array(dense2_weight_params, ctx=tvm.cpu(0))

module = runtime.create(graph, lib, ctx)
module.set_input("data", data)
module.set_input(**params)
module.run()

out_shape = (batch_size,10)
out = (module.get_output_data(8, tvm.nd.empty(out_shape))).asnumpy()
accury = ((checkTrainingResult(out,predict_out)) / float(batch_size))
print("accury is : %f" %accury)
