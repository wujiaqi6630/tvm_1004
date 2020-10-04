import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
import topi
import os
input(os.getpid())

out_channels = 1

data = relay.var("data", relay.TensorType((1,1,4,4), "float32"))
bias = relay.var("bias")
weight = relay.var("weight")
#bn_gamma = relay.var("bn_gamma")
#bn_beta = relay.var("bn_beta")
#bn_mmean = relay.var("bn_mean")
#bn_mvar = relay.var("bn_var")

print("----------TEST1----------")
#simple_net = relay.nn.addd(data = data,bias = bias)
simple_net = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3,3), channels=out_channels, padding=(0, 0))
#simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
print("----------TEST2----------")
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
print("----------TEST3----------")
data_shape = (1,1,4,4)
net, params = testing.create_workload(simple_net)
print("-----NET.ASTEXT----------")
print(net.astext(show_meta_data=False))

print("----------TEST4----------")
opt_level = 1
target = tvm.target.cuda()
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build_module.build(
        net, target, params=params)
        
print("----------TEST5----------")
ctx = tvm.gpu()
data = np.array([[[(1,2,3,4),(4,5,6,7),(7,8,9,0),(1,2,3,0.1232132131232311423141241241242141212421)]]]).astype("float32")
#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
print("----------INPUT_DATA----------")
print(data)
#params['bias'] = np.ones((3), dtype='int32')
print("----------TEST6----------")
module = runtime.create(graph, lib, ctx)
#set input and paramters
print("----------TEST7----------")
module.set_input("data", data)
module.set_input(**params)
print("----------INPUT_PARAMS----------")
print(params)
#run
print("----------TEST8----------")
module.run()
print("----------OUT_FLATTEN----------")
out_shape = (1,1,2,2)
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
print("----------TEST9----------")
print(out.flatten())
print("----------OUTPUT_PARAMS----------")
print(params)
params_kernel = np.array([[[(1,2,3),(4,5,6),(7,8,9)]]]).astype("float32")
params['p0'] = tvm.nd.array(params_kernel, ctx=tvm.cpu(0))
module.set_input("data", data)
#update params'value
module.set_input(**params)
module.run()
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
print("After update, params is:")
print(params)
print("final output is:")
tpl = out.flatten()
print(tpl)

A = tvm.placeholder((1, 1, 4, 4), name='A', dtype='float32')
B = topi.image.resize(A, (2, 2), "NCHW", False, "BILINEAR")
topi.image.unconv2d(out.flatten,out.flatten)
"""
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
# define loss function
loss = out.flatten()[0] + out.flatten()[1] + out.flatten()[2]
count = 0;
bias_tmp = params['p0'].asnumpy()
#p5_tmp = params['p5'].asnumpy()
print("Before update, params is:")
print(params)

while loss < 192:
    bias_tmp[0] += 5 
    bias_tmp[1] += 5
    bias_tmp[2] += 5
    bias_tmp[3] += 5
    #bias_tmp[4] += 5
    #bias_tmp[5] += 5
    pparam = np.array([(bias_tmp[0],bias_tmp[1],bias_tmp[2],bias_tmp[3])]).astype("float32")
    params['p0'] = tvm.nd.array(pparam, ctx=tvm.cpu(0))
   # p5_tmp[0] += 2  
   # p5_tmp[1] += 2
   # p5_tmp[2] += 2
   # p5param = np.array([(p5_tmp[0],p5_tmp[1],p5_tmp[2])]).astype("float32")
   # params['p5'] = tvm.nd.array(p5param, ctx=tvm.cpu(0))
    module.set_input("data", data)
    #update params'value
    module.set_input(**params)
    module.run()
    out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
    loss = out.flatten()[0] + out.flatten()[1] + out.flatten()[2]
    count += 1
print('After %d iterations, the training goal was achieved' %count)
print("After update, params is:")
print(params)
print("final output is:")
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
print(out.flatten())
"""

