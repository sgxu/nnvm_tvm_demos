# https://qiita.com/tkat0/items/28d1cc3b5c2831d86663

import tvm
import nnvm.compiler
import nnvm.testing

import os
import sys
from time import time

import keras
import numpy as np
from tvm.contrib import graph_runtime, rpc
from tvm.contrib import util, ndk, rpc_proxy
import nnvm.frontend
import nnvm.compiler

#exec_gpu = False
exec_gpu = True
dtype = np.float32
num_iter_for_time_test = 1
opt_level = 0

######################################################################
# Load pretrained keras model
# ----------------------------
# We load a pretrained resnet-50 classification model provided by keras.
keras_resnet50 = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=(224,224,3), classes=1000)
keras_resnet50.load_weights('../models/resnet50_weights.h5')

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.resnet50 import preprocess_input
img = Image.open('../images/cat.jpg').resize((224, 224))
#plt.imshow(img)
#plt.show()
# input preprocess
data = np.array(img)[np.newaxis, :].astype('float32')
data = preprocess_input(data).transpose([0, 3, 1, 2])
print('data', data.shape)

######################################################################
# connect to the proxy
# ------------------
# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"
print('RPC Connecting...')
remote = rpc.connect(proxy_host, proxy_port, key=key)
print('RPC Connected')

######################################################################
# Compile the model on NNVM
# --------------------------
# convert the keras model(NHWC layout) to NNVM format(NCHW layout).
sym, params = nnvm.frontend.from_keras(keras_resnet50)
# compile the model
arch = "arm64"
if exec_gpu:
    # Mobile GPU
    target = 'opencl'
    target_host = "llvm -target=%s-linux-android" % arch
    ctx = remote.cl(0)
else:
    # Mobile CPU
    target = "llvm -target=%s-linux-android" % arch
    target_host = None
    ctx = remote.cpu(0)

print('Build Graph...')
shape_dict = {'data': data.shape}
with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params, target_host=target_host)
#print("-------compute graph-------")
#print(graph.ir())
#print(graph.json())

######################################################################
# write graph.json, deploy.params, deploy.so
# ----------------------------------------
so_name = "deploy.so"
product_dir = "graph_model_lib"
#temp = util.tempdir()
#path_so = temp.relpath(so_name)
if not os.path.exists(product_dir):
    os.makedirs(product_dir)
path_so = os.path.abspath(product_dir)
path_so += "/" + so_name
print("newly generated android library path: %s" %path_so)
lib.export_library(path_so, ndk.create_shared)
# save graph as json format
graph_name = "graph.json"
path_graph = os.path.abspath(product_dir)
path_graph += "/" + graph_name
with open(path_graph, "w") as f:
        f.write(graph.json())
# save parameters
param_name = "deploy.params"
path_param = os.path.abspath(product_dir)
path_param += "/" + param_name
with open(path_param, 'wb') as f:
    f.write(nnvm.compiler.save_param_dict(params))
'''
print('show opencl kernel:')
print(lib.imported_modules[0].get_source())
'''
'''
######################################################################
# We can load the module back.
loaded_lib = tvm.module.load(path_so)
loaded_json = open(path_graph).read()
loaded_params = bytearray(open(path_param), "rb").read())
rmodule = graph_runtime.create(loaded_json, loaded_lib, tvm.gpu(0))
params = nnvm.compiler.load_param_dict(loaded_params)
# directly load from byte array
module.load_params(loaded_params)
module.run(x=x_np)
# get the first output
out = module.get_output(0, out=tvm.nd.empty(shape))
print(out.asnumpy())
'''

######################################################################
# upload lib to android
# --------------------------
print('DEPLOY: Shared Library Uploading...')
remote.upload(path_so)
rlib = remote.load_module(so_name)

######################################################################
# Execute on TVM
# --------------------------
# run on remote device
rmodule = graph_runtime.create(graph, rlib, ctx)
# set inputs
rmodule.set_input('data', tvm.nd.array(data.astype(dtype), ctx))
rmodule.set_input(**params)
#start = time()
print('Execute')
rmodule.run()
# get output
#output = tvm.nd.empty(out_shape, ctx=ctx)
#output = rmodule.get_output(0, output).asnumpy()
out_shape = (1000,)
tvm_out = rmodule.get_output(0, tvm.nd.empty(out_shape, ctx=ctx)).asnumpy()
top1_tvm = np.argmax(tvm_out)

synset_name = '../models/synset.txt'
with open(synset_name) as f:
    synset = eval(f.read())
print('NNVM top-1 id: {}, class name: {}'.format(top1_tvm, synset[top1_tvm]))
# confirm correctness with keras output
keras_out = keras_resnet50.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print('Keras top-1 id: {}, class name: {}'.format(top1_keras, synset[top1_keras]))

print('Benchmark')
ftimer = rmodule.module.time_evaluator("run", ctx, num_iter_for_time_test)
prof_res = ftimer()
print(prof_res, "sec")
