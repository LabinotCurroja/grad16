

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;




#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_kernel(const __half *a, const __half *b, float *c, int M, int N, int K) 
{
    // Calculate the starting row and column for this tile.
    int tile_row = blockIdx.y * WMMA_M;  // Starting row index of the tile
    int tile_col = blockIdx.x * WMMA_N;  // Starting column index of the tile

    // Declare the WMMA fragments.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    // Initialize the output fragment to zero.
    wmma::fill_fragment(cFrag, 0.0f);

    // Loop over the K dimension in steps of WMMA_K.
    for (int tile_k = 0; tile_k < K; tile_k += WMMA_K) 
    {
        const __half* aTile = a + tile_row * K + tile_k;
        const __half* bTile = b + tile_k * N + tile_col;
    
        // Load tiles from global memory.
        wmma::load_matrix_sync(aFrag, aTile, K);
        wmma::load_matrix_sync(bFrag, bTile, N);
    
        // Perform matrix multiplication.
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // Store the result back to global memory.
    float* cTile = c + tile_row * N + tile_col;
    wmma::store_matrix_sync(cTile, cFrag, N, wmma::mem_row_major);
}


extern "C" void matmul(half* A, half* B, float* C, int M, int N, int K) 
{
    //dim3 gridDim(1, 1);
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 blockDim(32, 1,1);
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize(); 
}
