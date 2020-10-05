import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tvm
from tvm.contrib import graph_runtime as runtime
from tvm.relay.backend import graph_runtime_codegen
from tvm import relay
from tvm.relay import testing
import topi
from timeit import default_timer as timer

#np.set_printoptions(threshold=np.nan) 
import sys
np.set_printoptions(threshold=np.sys.maxsize)

batch_size = 32
data = relay.var("data", relay.TensorType((batch_size,3,227,227), "float32"))
conv1_weight = relay.var("conv1_weight")
conv2_weight = relay.var("conv2_weight")
conv3_weight = relay.var("conv3_weight")
conv4_weight = relay.var("conv4_weight")
conv5_weight = relay.var("conv5_weight")
conv6_weight = relay.var("conv6_weight")
conv7_weight = relay.var("conv7_weight")
conv8_weight = relay.var("conv8_weight")

bias1 =  relay.var("bias", relay.TensorType((96,), "float32"))
bias2 =  relay.var("2bias")
bias3 =  relay.var("3bias")
bias4 =  relay.var("4bias")
bias5 =  relay.var("5bias")
bias6 =  relay.var("6bias")
bias7 =  relay.var("7bias")
bias8 =  relay.var("8bias")


simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(11,11), channels=96, strides=(4,4),padding=(0, 0))
simple_net = relay.nn.bias_add(data=simple_net, bias=bias1)
simple_net = relay.reshape(simple_net, (batch_size,96,55,55))
simple_net = relay.nn.relu(data=simple_net)
simple_net = relay.reshape(simple_net, (batch_size,96,55,55))
simple_net = relay.nn.lrn(data=simple_net,size=5, axis=1, bias=2, alpha=.00001, beta=0.75)
simple_net = relay.reshape(simple_net, (batch_size,96,55,55))
simple_net = relay.nn.max_pool2d(data=simple_net, pool_size=(3, 3), strides=(2, 2), padding=(0, 0), layout="NCHW")

simple_net = relay.nn.conv2d(data=simple_net, weight=conv2_weight, kernel_size=(5,5), channels=256, strides=(1,1),padding=(2, 2))
simple_net = relay.nn.bias_add(data=simple_net, bias=bias2)
simple_net = relay.reshape(simple_net, (batch_size,256,27,27))
simple_net = relay.nn.relu(data=simple_net)
simple_net = relay.reshape(simple_net, (batch_size,256,27,27))
simple_net = relay.nn.lrn(data=simple_net,size=5, axis=1, bias=2, alpha=.00001, beta=0.75)
simple_net = relay.reshape(simple_net, (batch_size,256,27,27))
simple_net = relay.nn.max_pool2d(data=simple_net, pool_size=(3, 3), strides=(2, 2), padding=(0, 0), layout="NCHW")

simple_net = relay.nn.conv2d(data=simple_net, weight=conv3_weight, kernel_size=(3,3), channels=384, strides=(1,1),padding=(1, 1))
simple_net = relay.nn.bias_add(data=simple_net, bias=bias3)
simple_net = relay.reshape(simple_net, (batch_size,384,13,13))
simple_net = relay.nn.relu(data=simple_net)

simple_net = relay.nn.conv2d(data=simple_net, weight=conv4_weight, kernel_size=(3,3), channels=384, strides=(1,1),padding=(1, 1))
simple_net = relay.nn.bias_add(data=simple_net, bias=bias4)
simple_net = relay.reshape(simple_net, (batch_size,384,13,13))
simple_net = relay.nn.relu(data=simple_net)

simple_net = relay.nn.conv2d(data=simple_net, weight=conv5_weight, kernel_size=(3,3), channels=256, strides=(1,1),padding=(1, 1))
simple_net = relay.nn.bias_add(data=simple_net, bias=bias5)
simple_net = relay.reshape(simple_net, (batch_size,256,13,13))
simple_net = relay.nn.relu(data=simple_net)
simple_net = relay.reshape(simple_net, (batch_size,256,13,13))
simple_net = relay.nn.lrn(data=simple_net, size=5, axis=1, bias=2, alpha=.00001, beta=0.75)
simple_net = relay.reshape(simple_net, (batch_size,256,13,13))
simple_net = relay.nn.max_pool2d(data=simple_net, pool_size=(3, 3), strides=(2, 2), padding=(0, 0), layout="NCHW")


simple_net = relay.reshape(simple_net, (batch_size,9216))
simple_net = relay.nn.dense(data=simple_net, weight=conv6_weight,units=1000)

simple_net = relay.nn.bias_add(data=simple_net, bias=bias6)
simple_net = relay.nn.relu(data=simple_net)



simple_net = relay.nn.dense(data=simple_net, weight=conv7_weight,units=1000)
simple_net = relay.nn.bias_add(data=simple_net, bias=bias7)
simple_net = relay.nn.relu(data=simple_net)

simple_net = relay.nn.dense(data=simple_net, weight=conv8_weight,units=1000)
simple_net = relay.nn.bias_add(data=simple_net, bias=bias8)

simple_net = relay.reshape(simple_net, (batch_size,1000))
simple_net = relay.nn.softmax(data=simple_net)


tic = timer()
node = relay.analysis.free_vars(simple_net)
simple_net = relay.Function(node, simple_net)
net, params = testing.create_workload(simple_net)
tg = "c"

with relay.build_config(opt_level=3):
    mod, _ = relay.optimize(net, tg, params)
    graph0, func0, params0 = graph_runtime_codegen.GraphRuntimeCodegen(None, tg).codegen(mod["main"])
    func=tvm.build(func0, tg, name="default_function")
toc = timer()

print("AlexNet compile on TVM time is : ", (toc-tic))
