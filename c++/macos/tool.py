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

exec_gpu = False
#exec_gpu = True
dtype = np.float32
num_iter_for_time_test = 1
opt_level = 0

######################################################################
# Load pretrained keras model
# ----------------------------
# We load a pretrained resnet-50 classification model provided by keras.
keras_resnet50 = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=(224,224,3), classes=1000)
keras_resnet50.load_weights('../../models/resnet50_weights.h5')

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.resnet50 import preprocess_input
img = Image.open('../../images/cat.jpg').resize((224, 224))
#plt.imshow(img)
#plt.show()
# input preprocess
data = np.array(img)[np.newaxis, :].astype('float32')
data = preprocess_input(data).transpose([0, 3, 1, 2])
print('data', data.shape)

######################################################################
# Compile the model on NNVM
# --------------------------
# convert the keras model(NHWC layout) to NNVM format(NCHW layout).
sym, params = nnvm.frontend.from_keras(keras_resnet50)
# compile the model
if exec_gpu:
    # Mobile GPU
    target = 'opencl'
    target_host = None
    ctx = tvm.cl(0)
else:
    # Mobile CPU
    target = 'llvm'
    target_host = None
    ctx = tvm.cpu(0)

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
print("newly generated library path: %s" %path_so)
lib.export_library(path_so)
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

if exec_gpu:
    print('show source code:')
    print(lib.imported_modules[0].get_source())


