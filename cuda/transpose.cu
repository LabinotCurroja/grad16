
#include <cuda_fp16.h>  // Include the CUDA half-precision header

#define TILE_SIZE 16  // Define the block size


__global__ void transpose_kernel(__half* input, __half* output, int rows, int cols) 
{
    // Declare shared memory for the block using __half
    __shared__ __half sharedTile[TILE_SIZE][TILE_SIZE];

    // Compute the global thread index
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data into shared memory (only if within bounds)
    if (x < cols && y < rows) {
        sharedTile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    // Synchronize threads to make sure all data is loaded into shared memory
    __syncthreads();

    // Write transposed data to global memory
    if (x < rows && y < cols) {
        output[y * rows + x] = sharedTile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void transpose_half(__half* input, __half* output, int rows, int cols) 
{

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}