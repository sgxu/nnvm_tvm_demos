# Specify the proxy host
export TVM_ANDROID_RPC_PROXY_HOST=0.0.0.0
# Specify the standalone Android C++ compiler
export TVM_NDK_CC=../android-toolchain/android-toolchain-arm64/bin/aarch64-linux-android-g++
python android_rpc_test.py
