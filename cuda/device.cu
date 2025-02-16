#include <cuda_runtime.h>
#include <stdio.h>

extern "C" 
{
    bool supportf16() 
    {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);

        if (error != cudaSuccess) 
        {
            return false;
        }

        if (deviceCount == 0) 
        {
            return false;
        }

        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) 
        {
            struct cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, deviceIndex);

            // Check if the GPU supports FP16 (Volta or newer)
            if (deviceProp.major >= 7 && deviceProp.minor >= 0) 
            {
                return true;  // FP16 is supported
            }
        }

        return false; 
    }

    bool is_gpu_available() 
    {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);

        if (error != cudaSuccess) 
        {
            return false;
        }

        if (deviceCount == 0) 
        {
            return false;
        }

        return true;
    }

    int cuda_malloc(void** devPtr, size_t size) 
    {
        cudaError_t status = cudaMalloc(devPtr, size);
        return status;
    }

    int cuda_free(void* devPtr) 
    {
        cudaError_t status = cudaFree(devPtr);
        return status;        
    }

    int cuda_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind)
    {
        cudaError_t status = cudaMemcpy(dst, src, size, kind);
        return status;
    }



}
