import os

os.environ["GLOG_minloglevel"] = "2"
import sys
import logging

logging.basicConfig(level=logging.ERROR)

import numpy as np
from google.protobuf import text_format
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2 as pb

import tvm
from tvm import relay
from tvm.contrib import util, graph_runtime
from tvm.relay.backend import graph_runtime_codegen

from tvm.contrib.download import download_testdata
input(os.getpid())

CURRENT_DIR = os.path.join(os.path.expanduser("~"), ".tvm_test_data", "caffe_test")

#######################################################################
def _run_tvm(data, proto_file, blob_file):
    """ Run caffe model by TVM according to .caffemodel and .prototxt"""
    init_net = pb.NetParameter()
    predict_net = pb.NetParameter()

    # load model
    with open(proto_file, "r") as f:
        text_format.Merge(f.read(), predict_net)
    # load blob
    with open(blob_file, "rb") as f:
        init_net.ParseFromString(f.read())

    shape_dict = dict()
    dtype_dict = dict()
    if isinstance(data, (tuple, list)):
        for idx, d in enumerate(data):
            shape_dict["data" + str(idx)] = d.shape
            dtype_dict["data" + str(idx)] = "float32"
    else:
        shape_dict = {"data": data.shape}
        dtype_dict = {"data": "float32"}

    #print("++++++++++++++++++++++++++++++++++++++++")
    net, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)

    tg = "dpu"
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    #mod, _ = relay.optimize(net, tg, params)
    with relay.build_config(opt_level=0):
        mod, _ = relay.optimize(net, tg, params)
        #lib = relay.build(mod, target=target, target_host=target_host, params=params)
        #graph, lib, params = relay.build_module.build(mod, target=tg, params=params)
        graph0, func0, params0 = graph_runtime_codegen.GraphRuntimeCodegen(None, tg).codegen(mod["main"])
        dtype = "float32"
        func=tvm.build(func0, tg, name="default_function")
    f = open('/home/wangjj/wujq/test/alexnetCode.c', 'w')
    print(func.get_source(), file = f)
    f.close()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

def _test_network(data, proto_file, blob_file):
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)


#######################################################################
# Alexnet
# -----------


def _test_alexnet(data):
    """ One iteration of Alexnet """
    mean_val = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean_val = np.reshape(mean_val, (1, 3, 1, 1))
    mean_val = np.tile(mean_val, (1, 1, 227, 227))
    data_process = data - mean_val
    data_process = data_process.astype(np.float32)

    proto_file_url = (
        "https://github.com/BVLC/caffe/raw/master/models/" "bvlc_alexnet/deploy.prototxt"
    )
    blob_file_url = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
    proto_file = download_testdata(proto_file_url, "alexnet.prototxt", module="model")
    blob_file = download_testdata(blob_file_url, "alexnet.caffemodel", module="model")
    _test_network(data_process, proto_file, blob_file)


def test_forward_Alexnet():
    """ Alexnet """
    data = np.random.randint(0, 256, size=(32, 3, 227, 227)).astype(np.float32)
    _test_alexnet(data)


#######################################################################

if __name__ == "__main__":
    # End to End
    test_forward_Alexnet()
