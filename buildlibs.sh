
echo "Building shared libraries..."


nvcc -shared -o bin/device.so -Xcompiler -fPIC cuda/device.cu
echo "-device.so built"


nvcc -shared -Xcompiler -fPIC -arch=sm_75 -O3 -o bin/matmul.so cuda/matmul.cu
echo "-matmul.so built"


nvcc -shared -Xcompiler -fPIC -arch=sm_75 -O3 -o bin/add.so cuda/add.cu
echo "-add.so built"


nvcc -shared -Xcompiler -fPIC -arch=sm_75 -O3 -o bin/transpose.so cuda/transpose.cu
echo "-transpose.so built"