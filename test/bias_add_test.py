import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
import os
input(os.getpid())

data = relay.var("data", relay.TensorType((1,2), "float32"))
bias = relay.var("bias")

simple_net = relay.nn.bias_add(data,bias)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (1,2)
net, params = testing.create_workload(simple_net)

print("----------NET.ASTEXT----------")
print(net.astext(show_meta_data=False))

opt_level = 1
target = tvm.target.cuda()
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build_module.build(
        net, target, params=params)

ctx = tvm.gpu()
#data = np.random.uniform(0, 9, size=data_shape).astype("int32")
#data = np.array([(1,2,3),(4,5,6)]).astype("float32")
data = np.array([(1,2)]).astype("float32")
#ias = np.array([(1)]).astype("float32")
module = runtime.create(graph, lib, ctx)
#set input and paramters
module.set_input("data", data)
#odule.set_input("bias", bias)
module.set_input(**params)
#run
module.run()
#get output
print("----------OUT_FLATTEN----------")
out_shape = (1,2)
out = (module.get_output(0, tvm.nd.empty(out_shape))).asnumpy()
print(out.flatten())

