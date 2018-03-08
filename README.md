# nnvm_tvm_demos
deploy nnvm tvm to android
we did test on huawei Mate 10 Pro.
this python script references to ttps://qiita.com/tkat0/items/28d1cc3b5c2831d86663

# how to build TVM Android RPC
refer to https://github.com/dmlc/tvm/tree/master/apps/android_rpc

## Build with OpenCL
This application does not link any OpenCL library unless you configure it to. In app/src/main/jni/make you will find JNI Makefile config config.mk. Copy it to app/src/main/jni and modify it.

```
cd apps/android_rpc/app/src/main/jni
cp make/config.mk .
```

below is my example:

```
#-------------------------------------------------------------------------------
#  Template configuration for compiling
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  ./build.sh
#
#-------------------------------------------------------------------------------
APP_ABI = arm64-v8a

APP_PLATFORM = android-17

# whether enable OpenCL during compile
USE_OPENCL = 1

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES = /Users/sgxu/Desktop/Android/OpenCL-Headers/opencl12

# the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = /Users/sgxu/Desktop/Android/libOpenCL.so
```

+ In which, ADD_C_INCLUDES is the standard OpenCL headers, you can download from: https://github.com/KhronosGroup/OpenCL-Headers
+ In which, ADD_LDLIBS is the mobile phone's opencl lib, You can use adb pull to get the file to your MacBook:

```
adb pull /system/vendor/lib64/libOpenCL.so ./
```
Now use NDK to generate standalone toolchain for your device. For my test device, I use following command:
```
cd /opt/android-ndk/build/tools/
./make-standalone-toolchain.sh --platform=android-24 --use-llvm --arch=arm64 --install-dir=/Users/sgxu/Desktop/Android/android-toolchain-arm64
```
# How Run this demo
## python
```
cd python
./run.sh
```
