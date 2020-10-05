import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
from tvm.relay.backend import graph_runtime_codegen
from tvm import relay
from tvm.relay import testing
import topi
#from time import time
import os
input(os.getpid())
#np.set_printoptions(threshold=np.nan) 
import sys
np.set_printoptions(threshold=np.sys.maxsize)

batch_size = 4
data = relay.var("data", relay.TensorType((batch_size,1000), "float32"))
#simple_net = relay.nn.softmax(data)

maxelem= relay.nn.softmaxMax(data)
#maxelem = relay.reshape(maxelem,(batch_size,))
sumelem  = relay.nn.softmaxSum(maxelem,data)
simple_net = relay.nn.softmaxDiv(sumelem,maxelem,data)


node = relay.analysis.free_vars(simple_net)
print("**************test1*************")
simple_net = relay.Function(node, simple_net)

net, params = testing.create_workload(simple_net)
print("----------TEST4----------")
tg = "dpu"
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
mod, _ = relay.optimize(net, tg, params)
graph0, func0, params0 = graph_runtime_codegen.GraphRuntimeCodegen(None, tg).codegen(mod["main"])
func=tvm.build(func0, tg, name="default_function")
print(func.get_source())
#print(tvm.lower(lib,[data, conv1_weight, simple_net], simple_mode=True))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

