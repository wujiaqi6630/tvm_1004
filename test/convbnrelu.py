import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
#from tvm.contrib.debugger import debug_runtime as graph_runtime
import os
input(os.getpid())
# Create a simple network
# from https://docs.tvm.ai/tutorials/relay_quick_start.html#sphx-glr-tutorials-relay-quick-start-py

# create a very simple network 
# It consists of convolution, batch normalization, and ReLU activation.

out_channels = 16
batch_size = 1

data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3,3), channels=out_channels, padding=(1, 1))
simple_net = relay.nn.relu(simple_net)
simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
simple_net = relay.nn.relu(simple_net)
#simple_net = relay.nn.relu(simple_net)
#simple_net = relay.relu_grad(simple_net,simple_net)
#simple_net = relay.Function(relay.ir_pass.free_vars(simple_net), simple_net)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 3, 224, 224)
net, params = testing.create_workload(simple_net)
#print(simple_net.astext(show_meta_data=False))
print("----------NET.ASTEXT----------")
print(net.astext(show_meta_data=False))

# build and run this network with cuda backend
# By setting the logging level to DEBUG, 
# the result of Relay graph compilation will be dumped as pseudo code
opt_level = 1
target = tvm.target.cuda()
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build_module.build(
        net, target, params=params)

ctx = tvm.gpu()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.create(graph, lib, ctx)
#set input and paramters
module.set_input("data", data)
module.set_input(**params)
#run
module.run()
#get output
print("----------OUT_FLATTEN----------")
out_shape = (batch_size, out_channels, 224, 224)
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
print(out.flatten()[0:100])
"""
#verify the result with cudnn
net, params = testing.create_workload(simple_net)
target = "cuda -libs=cudnn" # use cudnn for convolution
graph, lib, params = relay.build_module.build(
        net, target, params=params)
ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.create(graph, lib, ctx)
module.set_input(**params)
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cudnn = out.asnumpy()
print("----------CHECH----------")
tvm.testing.assert_allclose(out, out_cudnn, rtol=1e-5)
"""
