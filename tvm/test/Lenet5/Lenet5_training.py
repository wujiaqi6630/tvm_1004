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

f = open("./params_saved.txt", 'w+')

batch_size = 64
data = relay.var("data", relay.TensorType((batch_size,1,28,28), "float32"))
data_shape=(batch_size,1,28,28)

conv1_weight = relay.var("conv1_weight")
conv2_weight = relay.var("conv2_weight")
dense1_weight = relay.var("dense1_weight")
dense2_weight = relay.var("dense2_weight")

simple_net = relay.nn.conv2d(data=data, weight=conv1_weight, kernel_size=(5,5), channels=20, strides=(1,1),padding=(0, 0))
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

#####TRAIN#####
out_shape = (batch_size,10)
out = (module.get_output_data(8, tvm.nd.empty(out_shape))).asnumpy()
loss,output_gredient = topi.image.getLossValue("cross_entropy",predict_out,out)
print("Zero sum loss:")
print(loss)

begin = time()
epoch = 0
while (epoch < 256):
    if (loss > 150):
        learn_rate = 0.01
    elif (loss >= 100) and (loss < 150):
        learn_rate = 0.005
    elif (loss >= 10) and (loss < 50):
        learn_rate = 0.001
    elif (loss >= 1) and (loss < 10):
        learn_rate = 0.0005
    elif (loss >= 0) and (loss < 1):
        learn_rate = 0.0001
        
    iter = 0
    min_loss = 10000.0
    max_loss = 0.0
    while (iter < (32000 / batch_size)):
        tic1 = time()
        #for the 8_th layer(softmax)
        layer08_input_gradient = np.zeros(out_shape, dtype=np.float32)
        layer08_input_gradient = topi.image.softmax_gredient(output_gredient,out,layer08_input_gradient,predict_out,axis=1,lossFuncType="cross_entropy")
   
        #for the 7_th layer(fc)
        layer07_input_shape = (batch_size,500)
        layer07_param_shape = (10,500)
        layer07_input = (module.get_input_data(7, 0, tvm.nd.empty(layer07_input_shape))).asnumpy()
        layer07_param = (module.get_layer_params(7, 0, tvm.nd.empty(layer07_param_shape))).asnumpy()
        layer07_input_gradient = np.zeros(layer07_input_shape, dtype=np.float32)
        layer07_param_gradient = np.zeros(layer07_param_shape, dtype=np.float32)
        layer07_param,layer07_input_gradient = topi.image.dense_gredient(layer08_input_gradient,layer07_input,layer07_param,layer07_input_gradient,layer07_param_gradient,learning_rate=learn_rate)

        #for the 6_th layer(relu):
        layer06_input_shape = (batch_size,500)
        layer06_input = (module.get_input_data(6, 0, tvm.nd.empty(layer06_input_shape))).asnumpy()
        layer06_input_gradient = np.zeros(layer06_input_shape, dtype=np.float32)
        layer06_input_gradient = topi.image.relu_gredient(layer07_input_gradient,layer06_input,layer06_input_gradient)

        #for the 5_th layer(fc)
        layer05_input_shape = (batch_size,800)
        layer05_param_shape = (500,800)
        layer05_input = (module.get_input_data(5, 0, tvm.nd.empty(layer05_input_shape))).asnumpy()
        layer05_param = (module.get_layer_params(5, 0, tvm.nd.empty(layer05_param_shape))).asnumpy()
        layer05_input_gradient = np.zeros(layer05_input_shape, dtype=np.float32)
        layer05_param_gradient = np.zeros(layer05_param_shape, dtype=np.float32)
        layer05_param,layer05_input_gradient = topi.image.dense_gredient(layer06_input_gradient,layer05_input,layer05_param,layer05_input_gradient,layer05_param_gradient,learning_rate=learn_rate)

        #for the 4_th layer(flatten)
        layer04_input_shape = (batch_size,50,4,4)
        layer04_input_gradient = np.zeros(layer04_input_shape, dtype=np.float32)
        layer04_input_gradient = topi.image.batch_flatten_gredient(layer05_input_gradient,layer04_input_gradient)

        #for the 3_th layer(max_pooling)
        layer03_input_shape = (batch_size,50,8,8)
        layer03_output_shape = (batch_size,50,4,4)
        layer03_input = (module.get_input_data(3, 0, tvm.nd.empty(layer03_input_shape))).asnumpy()
        layer03_output = (module.get_output_data(3, tvm.nd.empty(layer03_output_shape))).asnumpy()
        layer03_input_gradient = np.zeros(layer03_input_shape, dtype=np.float32)
        layer03_input_gradient = topi.image.pooling_gredient(layer04_input_gradient,layer03_input,layer03_output,layer03_input_gradient,pool_size=(2, 2),strides=(2,2),padding=(0,0),poolType="max")

        #for the 2_th layer(conv_output2):
        layer02_input_shape = (batch_size,20,12,12)
        layer02_param_shape = (50,20,5,5)
        layer02_input = (module.get_input_data(2, 0, tvm.nd.empty(layer02_input_shape))).asnumpy()
        layer02_param = (module.get_layer_params(2, 0, tvm.nd.empty(layer02_param_shape))).asnumpy()
        layer02_input_gradient = np.zeros(layer02_input_shape, dtype=np.float32)
        layer02_param_gradient = np.zeros(layer02_param_shape, dtype=np.float32)
        layer02_param,layer02_input_gradient = topi.image.conv2d_gredient(layer03_input_gradient,layer02_input,layer02_param,layer02_input_gradient,layer02_param_gradient,strides=(1,1),padding=(0,0),learning_rate=learn_rate)

        #for the 1_th layer(max_pooling)
        layer01_input_shape = (batch_size,20,24,24)
        layer01_output_shape = (batch_size,20,12,12)
        layer01_input = (module.get_input_data(1, 0, tvm.nd.empty(layer01_input_shape))).asnumpy()
        layer01_output = (module.get_output_data(1, tvm.nd.empty(layer01_output_shape))).asnumpy()
        layer01_input_gradient = np.zeros(layer01_input_shape, dtype=np.float32)
        layer01_input_gradient = topi.image.pooling_gredient(layer02_input_gradient,layer01_input,layer01_output,layer01_input_gradient,pool_size=(2, 2),strides=(2,2),padding=(0,0),poolType="max")

        #for the 0-th layer(conv_output1):
        layer00_input_shape = (batch_size,1,28,28)
        layer00_param_shape = (20,1,5,5)
        layer00_input = (module.get_input_data(0, 0, tvm.nd.empty(layer00_input_shape))).asnumpy()
        layer00_param = (module.get_layer_params(0, 0, tvm.nd.empty(layer00_param_shape))).asnumpy()
        layer00_input_gradient = np.zeros(layer00_input_shape, dtype=np.float32)
        layer00_param_gradient = np.zeros(layer00_param_shape, dtype=np.float32)
        layer00_param,layer00_input_gradient = topi.image.conv2d_gredient(layer01_input_gradient,layer00_input,layer00_param,layer00_input_gradient,layer00_param_gradient,strides=(1,1),padding=(0,0),learning_rate=learn_rate)

        params[paramsName_pair['conv1_weight']] = tvm.nd.array(layer00_param, ctx=tvm.cpu(0))
        params[paramsName_pair['conv2_weight']] = tvm.nd.array(layer02_param, ctx=tvm.cpu(0))
        params[paramsName_pair['dense1_weight']] = tvm.nd.array(layer05_param, ctx=tvm.cpu(0))
        params[paramsName_pair['dense2_weight']] = tvm.nd.array(layer07_param, ctx=tvm.cpu(0))

        iter += 1
        data = images[iter*batch_size:(iter+1)*batch_size]
        predict_out = labels[iter*batch_size:(iter+1)*batch_size]
        module.set_input("data", data)
        module.set_input(**params)
        module.run()
        tic2 = time()
        out = (module.get_output_data(8, tvm.nd.empty(out_shape))).asnumpy()
        loss,output_gredient = topi.image.getLossValue("cross_entropy",predict_out,out)
        print("##########################################################################################")
        print("Compelte: epoch is %d ; iteration is %d ; time is %f ; sumTime is %f ." %(epoch,(iter-1),(tic2-tic1),(tic2-begin)))
        print("loss : %f" %(loss))
        print("##########################################################################################")
        print("\n\n")
        if (loss < min_loss):
            min_loss = loss
        if (loss > max_loss):
            max_loss = loss
        if (loss > 500):
            exit()    
        
    print("######################################################",file=f)
    print("The %d-th's params:. " %(epoch),file=f)
    print("loss range from  %f  to  %f . "%(min_loss,max_loss),file=f)
    if (epoch == 128 or epoch % 10 == 0):
        print("***********************************************",file=f)
        print("The %d-th's params:. " %(epoch),file=f)
        str=""
        for i in range (epoch+1):
            str += "@"
        print(str,file=f)
        print(params,file=f)
        print((str+"!"),file=f)
        print("***********************************************",file=f)
    print("######################################################",file=f)
    print("\n\n",file=f)
    epoch += 1
    print("Begin %d epoch. "%(epoch))
    
end = time()   
print("Training times : ")
print(end-begin)
