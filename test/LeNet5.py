import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
import topi
import os
input(os.getpid())

data = relay.var("data", relay.TensorType((1,1,8,8), "float32"))
conv1_weight = relay.var("conv1_weight")
conv2_weight = relay.var("conv2_weight")

print("----------TEST1----------")
simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(3,3), channels=1, padding=(0, 0))
simple_net = relay.nn.relu(simple_net)
simple_net = relay.nn.avg_pool2d(simple_net, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
simple_net = relay.nn.relu(simple_net)
simple_net = relay.nn.conv2d(simple_net, weight=conv2_weight, kernel_size=(2,2), channels=1, padding=(0, 0))
print("----------TEST2----------")
node = relay.analysis.free_vars(simple_net)
simple_net = relay.Function(node, simple_net)
print("----------TEST3----------")
net, params = testing.create_workload(simple_net)
print("-----NET.ASTEXT----------")
print(net.astext(show_meta_data=False))

print("----------TEST4----------")
opt_level = 0
target = tvm.target.cuda()
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(net, target, params=params)
 
print("----------TEST5----------")
ctx = tvm.gpu()
#data = np.array([[[(1,2,3,4),(2,3,4,1),(3,4,1,2),(4,1,2,3)]]]).astype("float32")
data = np.array([[[(1,2,3,4,5,6,7,8),(2,3,4,5,6,7,8,9),(3,4,5,6,7,8,9,0),(4,5,6,7,8,9,0,-1),(5,6,7,8,9,0,-1,-2),(6,7,8,9,0,-1,-2,-3),(7,8,9,0,-1,-2,-3,-4),(8,9,0,0,0,0,0,0)]]]).astype("float32")
#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
print("----------INPUT_DATA----------")
print(data)
print("----------TEST6----------")
module = runtime.create(graph, lib, ctx)
print("----------TEST7----------")
module.set_input("data", data)
module.set_input(**params)
print("----------INPUT_PARAMS----------")
print(params)
print("----------TEST8----------")
module.run()
print("----------OUT_FLATTEN----------")
out_shape = (1,1,2,2)
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
print("----------TEST9----------")
print(out)
#choose loss_function
loss_type = "quadratic"
#predict_out = np.array([[[(1,2),(3,4)]]]).astype("float32")
predict_out = np.array([[[(12.0)]]]).astype("float32")
learning_rate = 0.5
loss_value = topi.image.getLossValue(loss_type,predict_out,out)
#for the 4-th layer:
layer_output_shape = (1,1,1,1)
layer_output = (module.get_output_data(4, tvm.nd.empty(layer_output_shape))).asnumpy()
print("----------TEST10----------")
print(layer_output)
print("----------TEST11----------")
layer_input_shape = (1,1,3,3)
layer_input = (module.get_input_data(4, tvm.nd.empty(layer_input_shape))).asnumpy()
print(layer_input)
layer_params01_shape = (1,1,3,3)
layer_params01 = (module.get_layer_params(4, 0, tvm.nd.empty(layer_params01_shape))).asnumpy()
print("----------TEST12----------")
print(layer_params01)
#layer5_params = [layer_params01]
layer_params00 = topi.image.unconv2d(loss_value, learning_rate, layer_input, layer_output, layer_params01)
print("----------TEST13----------")
print(layer_params00)
