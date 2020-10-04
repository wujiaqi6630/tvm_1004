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

np.set_printoptions(threshold=np.nan) 

batch_size = 5
data = relay.var("data", relay.TensorType((batch_size,1,24,24), "float32"))
data_shape=(batch_size,1,24,24)

conv1_weight = relay.var("conv1_weight")
conv2_weight = relay.var("conv2_weight")
dense1_weight = relay.var("dense1_weight")
dense2_weight = relay.var("dense2_weight")

#simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(5,5), channels=20, strides=(1,1),padding=(0, 0))
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

images = topi.image.load_train_images()
labels = topi.image.load_train_labels()
data = images[0:batch_size]
predict_out = labels[0:batch_size]

module = runtime.create(graph, lib, ctx)
module.set_input("data", data)
module.set_input(**params)
paramsName_pair = relay.build_module.getParamsNamePair()
module.run()
"""

#####TRAIN#####
out_shape = (batch_size,10)
out123 = (module.get_output_data(8, tvm.nd.empty(out_shape))).asnumpy()
print(out123)
out = out123.tolist()
print(out)

for i in range(len(out)):
    print(out[i].index(max(out[i])))
"""
