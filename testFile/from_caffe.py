"""
Caffe testcases
====================
This article is a test script to test Caffe operator with Relay.
"""
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
from tvm.contrib.download import download_testdata

CURRENT_DIR = os.path.join(os.path.expanduser("~"), ".tvm_test_data", "caffe_test")

#######################################################################
# Generic functions for TVM & Caffe
def _run_caffe(data, proto_file, blob_file):
    """ Run caffe model by Caffe according to .caffemodel and .prototxt"""
    net = caffe.Net(proto_file, blob_file, caffe.TEST)
    if isinstance(data, (list, tuple)):
        for idx, d in enumerate(data):
            net.blobs["data" + str(idx)].data[...] = d
    else:
        net.blobs["data"].data[...] = data
    out = net.forward()

    caffe_output = list()
    for i in range(len(out.keys())):
        if "output" + str(i) not in out.keys():
            caffe_output.clear()
            return list(out.values())
        caffe_output.append(out["output" + str(i)])
    return caffe_output


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

    mod, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)

    target = "llvm"
    target_host = "llvm"

    ctx = tvm.cpu(0)
    #with tvm.transform.PassContext(opt_level=3):
    with relay.build_config(opt_level=0):
        #lib = relay.build(mod, target=target, target_host=target_host, params=params)
        _, lib, _ = relay.build_module.build(mod, target=target, target_host=target_host, params=params)
    dtype = "float32"
    m = graph_runtime.GraphModule(lib["default"](ctx))
    if isinstance(data, (tuple, list)):
        for idx, d in enumerate(data):
            m.set_input("data" + str(idx), tvm.nd.array(d.astype(dtype)))
    else:
        m.set_input("data", tvm.nd.array(data.astype(dtype)))
    # execute
    m.run()
    tvm_output = list()
    # get outputs
    for i in range(m.get_num_outputs()):
        tvm_output.append(m.get_output(i).asnumpy())
    return tvm_output


def _compare_caffe_tvm(caffe_out, tvm_out, is_network=False):
    for i in range(len(caffe_out)):
        if is_network:
            caffe_out[i] = caffe_out[i][:1]
        tvm.testing.assert_allclose(caffe_out[i], tvm_out[i], rtol=1e-5, atol=1e-5)


def _test_op(data, func_op, op_name, **kwargs):
    """ Single op testing pipline. """
    shape_list = list()
    if isinstance(data, (list, tuple)):
        n = _miso_op(data, func_op, **kwargs)
        for d in data:
            shape_list.extend(list(d.shape))
    else:
        output_num = 1
        if "ntop" in kwargs.keys():
            output_num = kwargs["ntop"]
        if output_num == 1:
            n = _siso_op(data, func_op, **kwargs)
        else:
            n = _simo_op(data, func_op, **kwargs)
        shape_list = list(data.shape)

    # obtain the .caffemodel file and .prototxt file
    (proto_file, blob_file, solver_file) = _gen_filename_str(op_name, shape_list, **kwargs)
    _gen_model_files(n, proto_file, blob_file, solver_file)
    # run model in Caffe
    caffe_out = _run_caffe(data, proto_file, blob_file)
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)
    _compare_caffe_tvm(caffe_out, tvm_out)


def _test_network(data, proto_file, blob_file):
    # run model in Caffe
    caffe_out = _run_caffe(data, proto_file, blob_file)
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)
    _compare_caffe_tvm(caffe_out, tvm_out, is_network=True)


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
    data = np.random.randint(0, 256, size=(1, 3, 227, 227)).astype(np.float32)
    _test_alexnet(data)


#######################################################################

if __name__ == "__main__":
    # End to End
    test_forward_Alexnet()
