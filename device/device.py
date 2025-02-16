

import ctypes

cuda_lib = ctypes.CDLL("./bin/device.so")

# Specify the return type of the function @supportf16 && @is_gpu_available 
cuda_lib.supportf16.restype       = ctypes.c_bool
cuda_lib.is_gpu_available.restype = ctypes.c_bool


cuda_lib.cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda_lib.cuda_malloc.restype  = ctypes.c_int

cuda_lib.cuda_free.argtypes   = [ctypes.c_void_p]
cuda_lib.cuda_free.restype    = ctypes.c_int

cuda_lib.cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cuda_lib.cuda_memcpy.restype  = ctypes.c_int


cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2



def fp16_support():
    return cuda_lib.supportf16()

def is_gpu_available():
    return cuda_lib.is_gpu_available()

def cuda_malloc(data, size):
    """Wrapper for cudaMalloc."""
    status = cuda_lib.cuda_malloc(data, size)
    return status

def cuda_free(data):
    """Wrapper for cudaFree."""
    status = cuda_lib.cuda_free(data)
    return status

def cuda_memset(dst, src, size, kind):
    status = cuda_lib.cuda_memcpy(dst, src, size, cudaMemcpyHostToDevice)
    return status