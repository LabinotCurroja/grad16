

import ctypes

cuda_lib           = ctypes.CDLL("./bin/device.so")
cuda_kernels       = ctypes.CDLL("./bin/matmul.so")
cuda_add           = ctypes.CDLL("./bin/add.so")
cuda_transpose     = ctypes.CDLL("./bin/transpose.so")

""" bindings for CUDA functions """

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




""" arithmetics for CUDA kernels - move this to another file later for better organization """
cuda_kernels.matmul.argtypes = [
    ctypes.c_void_p,  # a (GPU pointer)
    ctypes.c_void_p,  # b (GPU pointer)
    ctypes.c_void_p,  # c (GPU pointer)
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int      # K
]
cuda_kernels.matmul.restype = None  # No return value


cuda_kernels.matmul_mini.argtypes = [
    ctypes.c_void_p,  # a (GPU pointer)
    ctypes.c_void_p,  # b (GPU pointer)
    ctypes.c_void_p,  # c (GPU pointer)
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int      # K
]
cuda_kernels.matmul_mini.restype = None  # No return value


cuda_add.add.argtypes = [
    ctypes.c_void_p,  # a (GPU pointer)
    ctypes.c_void_p,  # b (GPU pointer)
    ctypes.c_void_p,  # c (GPU pointer)
    ctypes.c_int,     # M
    ctypes.c_int      # N
]
cuda_add.add.restype = None  # No return value



cuda_transpose.transpose.argtypes = [
    ctypes.c_void_p,  # a (GPU pointer)
    ctypes.c_void_p,  # b (GPU pointer)
    ctypes.c_int,     # rows
    ctypes.c_int      # cols
]
cuda_transpose.transpose.restype = None  # No return value




cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


""" wrappers for CUDA functions """

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

def matmul(a, b, c, M, N, K):

    if (M % 16 == 0) and (N % 16 == 0) and (K % 16 == 0):
        cuda_kernels.matmul(a, b, c, M, N, K)
    else:
        cuda_kernels.matmul_mini(a, b, c, M, N, K)
    

def add(a, b, c, M, N):
    cuda_add.add(a, b, c, M, N)

def transpose(a, b, rows, cols):
    cuda_transpose.transpose(a, b, rows, cols)