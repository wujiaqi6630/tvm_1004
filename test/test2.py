from numba import njit
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tvm
import numpy as np
import tvm.contrib
from tvm.contrib import graph_runtime as runtime

from tvm import relay
from tvm.relay import testing
from tvm.relay.backend import graph_runtime_codegen
import topi
import os
input(os.getpid())

data = relay.var("data", relay.TensorType((2,4), "float32"))
data_shape=(2,4)
dense1_weight = relay.var("dense1_weight")
dense2_weight = relay.var("dense2_weight")
dense3_weight = relay.var("dense3_weight")

#simple_net = relay.nn.dense(data, dense1_weight,units=5)
#simple_net = relay.nn.dense(simple_net, dense2_weight,units=4)
dense_net = relay.nn.dense(data, dense3_weight,units=3)
simple_net = relay.nn.softmax(dense_net,1)
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST1%%%%%%%%%%%%%%%%%%%%")
node = relay.analysis.free_vars(simple_net)
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST2%%%%%%%%%%%%%%%%%%%%")
simple_net = relay.Function(node, simple_net)
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST3%%%%%%%%%%%%%%%%%%%%")
net, params = testing.create_workload(simple_net)
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST4%%%%%%%%%%%%%%%%%%%%")
opt_level = 0
#target = tvm.target.cuda()
#ctx = tvm.gpu()
target = 'llvm'
mod, _ = relay.optimize(net, target, params)
target_host = "cpu"

ctx = tvm.cpu(0)
print("%%%%%%%%%s%%%%%%%%%%%%%%%TEST5%%%%%%%%%%%%%%%%%%%%")
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(net, target, params=params)
    #libs = tvm.build(s, [A, B, C], target_host=tgt_host,name='DPUGemm')
#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
print("^^^^^^^^^^^^^^^^^^GET_SOURCE^^^^^^^^^^^^^^^^^^^^^^^^")
#print(lib.get_source())
print("^^^^^^^^^^^^^^^^^^GET_SOURCE^^^^^^^^^^^^^^^^^^^^^^^^")
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST6%%%%%%%%%%%%%%%%%%%%")

grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
grc.codegen(mod["main"])
#graph_json, lowered_func, params = graph_runtime_codegen.GraphRuntimeCodegen(simple_net)
#graphRunCod = runtimeCodegen.GraphRuntimeCodegen(graph,target)
#graph_json, lowered_func, params = graphRunCod.codegen(simple_net)

#print(graph)
temp_param = np.array(
[[-1.,  1.,  0.,  2.],
 [1.,  0.,  -1.,  1.],
 [ 0., -1.,  1., -1.]]).astype("float32")


data = np.array(
[[-1.,  1.,  0.,  2.],
 [ 0., -1.,  1., -1.]]).astype("float32")
   
#params['p0'] = tvm.nd.array(temp_param, ctx=tvm.cpu(0))
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST7%%%%%%%%%%%%%%%%%%%%")
module = runtime.create(graph, lib, ctx)
print("%%%%%%%%%%%%%%%%%%%%%%%%TEST8%%%%%%%%%%%%%%%%%%%%")
module.set_input("data", data)
module.set_input(**params)
print("%%%%%%params%%%%%%%")
print(params)
module.run()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test9%%%%%%%%%%%%%%%%%%%%%%%")
out_shape = (2,3)
out = (module.get_output(5712, tvm.nd.empty(out_shape))).asnumpy()
print("----------TEST10----------")
print(out)

