

nvcc -shared -o bin/device.so -Xcompiler -fPIC cuda/device.cu
#nvcc -shared -o bin/matmul.so -Xcompiler -fPIC cuda/matmul.cu