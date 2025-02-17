

nvcc -shared -o bin/device.so -Xcompiler -fPIC cuda/device.cu
#nvcc -shared -o bin/matmul.so -Xcompiler -fPIC cuda/matmul.cu
nvcc -shared -Xcompiler -fPIC -arch=sm_75 -O3 -o bin/matmul.so cuda/matmul.cu
nvcc -shared -Xcompiler -fPIC -arch=sm_75 -O3 -o bin/add.so cuda/add.cu