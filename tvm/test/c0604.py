#wujiaqi
from numba import njit
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tvm
from tvm.contrib import graph_runtime as runtime
from tvm.relay.backend import graph_runtime_codegen
from tvm import relay
from tvm.relay import testing
import topi
import os
from time import time
input(os.getpid())
np.set_printoptions(threshold=np.nan) 

batch_size = 2
data = relay.var("data", relay.TensorType((batch_size,3,4,4), "float32"))
conv1_weight = relay.var("conv1_weight")
conv2_weight = relay.var("conv2_weight")
dense1_weight = relay.var("dense1_weight")
dense2_weight = relay.var("dense2_weight")

simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(2,2), channels=2, strides=(2,2),padding=(1, 1))
#simple_net = relay.nn.max_pool2d(simple_net,pool_size=(2, 2),strides=(2, 2),padding=(1, 1))
#simple_net = relay.nn.batch_flatten(simple_net)
#simple_net = relay.nn.dense(simple_net, dense1_weight,units=10)
#simple_net = relay.nn.relu(simple_net)
#simple_net = relay.nn.softmax(simple_net,1)

node = relay.analysis.free_vars(simple_net)
print("**************test1*************")
simple_net = relay.Function(node, simple_net)
print("**************test2*************")
net, params_tmp = testing.create_workload(simple_net)
target="c"
print("**************test3*************")
mod, _ = relay.optimize(net, target, params_tmp)
print("**************test4*************")
grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
print("**************test5*************")
graph0, func0, params0 = grc.codegen(mod["main"])
print(func0)
print("**************test6*************")
func=tvm.build(func0, target, name="default_function")
print("**************test7*************")
print(func.get_source())
