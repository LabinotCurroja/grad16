

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_T 16


// WMMA kernel: C = A * B
__global__ void matmul_kernel(const __half *a, const __half *b, float *c, int M, int N, int K) 
{
    int tile_row = blockIdx.y;  // tile row index
    int tile_col = blockIdx.x;  // tile col index

    // Declare the WMMA fragments.
    wmma::fragment<wmma::matrix_a, WMMA_T, WMMA_T, WMMA_T, __half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_T, WMMA_T, WMMA_T, __half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_T, WMMA_T, WMMA_T, float> cFrag;

    // Initialize the output fragment to zero.
    wmma::fill_fragment(cFrag, 0.0f);

    // For simplicity, assume K == WMMA_K so there is only one iteration.
    for (int tile_k = 0; tile_k < K; tile_k += WMMA_T) 
    {
        const __half* aTile = a + tile_row * WMMA_T * K + tile_k;
        const __half* bTile = b + tile_k * N + tile_col * WMMA_T;
    
        // Load tiles from global memory
        wmma::load_matrix_sync(aFrag, aTile, K);
        wmma::load_matrix_sync(bFrag, bTile, N);
    
        // Perform matrix multiplication
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }
    

    // Store the result back to global memory.
    float* cTile = c + tile_row * WMMA_T * N + tile_col * WMMA_T;
    wmma::store_matrix_sync(cTile, cFrag, N, wmma::mem_row_major);
}


extern "C" void matmul(half* A, half* B, float* C, int M, int N, int K) 
{
    dim3 gridDim(1, 1);
    dim3 blockDim(32, 1,1);
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
