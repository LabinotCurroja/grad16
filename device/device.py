

import ctypes

cuda_lib = ctypes.CDLL("./bin/device.so")

# Specify the return type of the function @supportf16 && @is_gpu_available 
cuda_lib.supportf16.restype       = ctypes.c_bool
cuda_lib.is_gpu_available.restype = ctypes.c_bool

# Specify the return type of the function @cuda_malloc
cuda_lib.cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda_lib.cuda_malloc.restype  = ctypes.c_int

# Specify the argument types of the function @cuda_free
cuda_lib.cuda_free.argtypes   = [ctypes.c_void_p]
cuda_lib.cuda_free.restype    = ctypes.c_int

# Specify the argument types of the function @cuda_memcopy
cuda_lib.cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cuda_lib.cuda_memcpy.restype  = ctypes.c_int


# Specify the return type of the function @gpu_memory_usage
cuda_lib.gpu_memory_usage.restype = ctypes.c_float


# Specify the return type of the function @gpu_memory_total
cuda_lib.gpu_memory_total.restype = ctypes.c_float

# Specify the return type of the function @gpu_memory_free
cuda_lib.gpu_memory_free.restype = ctypes.c_float

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

def cuda_memset(dst, src, size):
    status = cuda_lib.cuda_memcpy(dst, src, size, cudaMemcpyHostToDevice)
    return status

def cuda_memcopy(dst, src, size):
    status = cuda_lib.cuda_memcpy(dst, src, size, cudaMemcpyDeviceToHost)
    return status

def gpu_memory_allocated():
    return cuda_lib.gpu_memory_usage()

def gpu_memory_total():
    return cuda_lib.gpu_memory_total()

def gpu_memory_available():
    return cuda_lib.gpu_memory_free()