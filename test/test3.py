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
import topi
import os
input(os.getpid())

data = relay.var("data", relay.TensorType((2,2,7,7), "float32"))
data_shape=(2,2,7,7)
conv1_weight = relay.var("conv1_weight")

simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(2,2), channels=2, strides=(2,2),padding=(0, 0))
simple_net = relay.nn.relu(simple_net)

node = relay.analysis.free_vars(simple_net)
simple_net = relay.Function(node, simple_net)
net, params = testing.create_workload(simple_net)

opt_level = 0
target = tvm.target.cuda()
with relay.build_config(opt_level=opt_level):
    graph, lib, params = relay.build_module.build(net, target, params=params)
ctx = tvm.gpu()
#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

temp_param = np.array(
[[[[1.,  1.],
   [ 0., -1.]],

  [[1.,  0.],
   [ -1., 1.]]],

   [[[0.,  1.],
   [ 1., 0.]],

  [[1.,  0.],
   [ 0., -1.]]]]).astype("float32")

data = np.array(
[[[[-1.,  1.,  0.,  2.,  0.,  0., -2.],
   [ 0.,  1.,  0., -2., -1.,  1., -1.],
   [ 1.,  2., -1.,  0.,  2.,  2., -1.],
   [-1., -1.,  2.,   0,   1,   1,  -1],
   [ 1.,   1,   0,  -2,  1.,   0, 1.0],
   [-1.,  2.,   0,  1.,  2., -2.,  0.],
   [ 0., -1.,  1., -1.,  2.,  2.,  0.]],

  [[1.,  0.,  0.,  1.,  0.,  1., -1.],
   [ 0.,  -1.,  0., 0., -1.,  1., 1.],
   [ 1.,  1., -1.,  0.,  2.,  2., -1.],
   [1., 0.,  -2.,   0,   1,   1,  -1],
   [ 1.,   1,   0,  -2,  1.,   0, 1.0],
   [-1.,  0.,   0,  -1.,  2., 0.,  0.],
   [ -1., -1.,  1., -1.,  1.,  1.,  0.]]],

  [[[-1.,  -1.,  0.,  1.,  0.,  -1., -2.],
   [ 0.,  1.,  0., -2., -1.,  1., -1.],
   [ -1.,  2., 1.,  0.,  1.,  2., 1.],
   [-1., -1.,  2.,   0,   1,   1,  -1],
   [ 1.,   -1,   0,  0,  1.,   0, -1.],
   [-1.,  2.,   0,  1.,  2., -2.,  0.],
   [ 2., -1.,  1., -1.,  1.,  0.,  1.]],

  [[1.,  -1.,  2.,  2.,  0.,  1., -1.],
   [ -2.,  1.,  0., -2., -1.,  1., 0.],
   [ 1.,  2., -1.,  0.,  0.,  2., -1.],
   [-1., -1.,  2.,   0,   1,   1,  -1],
   [ 0.,   1,   -2,  0,  1.,   0, -1.],
   [-1.,  2.,   0,  1.,  2., -2.,  1.],
   [ 0., -1.,  0., -1.,  -2.,  -1.,  1.]]]]).astype("float32")
   
params['p0'] = tvm.nd.array(temp_param, ctx=tvm.cpu(0))
module = runtime.create(graph, lib, ctx)
module.set_input("data", data)
module.set_input(**params)
print("%%%%%%params%%%%%%%")
print(params)
module.run()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test8%%%%%%%%%%%%%%%%%%%%%%%")
out_shape = (2,2,3,3)
out = (module.get_output(5712, tvm.nd.empty(out_shape))).asnumpy()
print("----------TEST9----------")
print(out)

batch, in_channel, in_height, in_width = data.shape
print(batch, in_channel, in_height, in_width)
