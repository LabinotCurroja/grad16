

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


__global__ void add_kernel(const __half *a, const __half *b, float *c, int M, int N)
{
    // Calculate the row and column index of the current thread.
    int row = blockIdx.y * WMMA_M + threadIdx.y;
    int col = blockIdx.x * WMMA_N + threadIdx.x;

    // Declare shared memory for blocks of a and b
    __shared__ __half shared_a[WMMA_M][WMMA_K];
    __shared__ __half shared_b[WMMA_K][WMMA_N];

    // Ensure we don't go out of bounds
    if (row < M && col < N)
    {
        // Load data into shared memory
        shared_a[threadIdx.y][threadIdx.x] = a[row * N + col];
        shared_b[threadIdx.y][threadIdx.x] = b[row * N + col];

        // Synchronize threads in the block
        __syncthreads();

        // Perform the element-wise addition using Tensor Cores
        if (row < M && col < N)
        {
            c[row * N + col] = __half2float(shared_a[threadIdx.y][threadIdx.x]) + __half2float(shared_b[threadIdx.y][threadIdx.x]);
        }
    }
}


extern "C" void add(half* A, half* B, float* C, int M, int N)
{
    dim3 blockDim(WMMA_M, WMMA_N);  // Tile size, based on Tensor Core usage
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);  // Grid size based on tensor dimensions
    
    add_kernel<<<gridDim, blockDim>>>(A, B, C, M, N);
    cudaDeviceSynchronize();

}