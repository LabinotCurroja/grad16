import struct
import ctypes
from device.device import cuda_malloc, cuda_free, cuda_memset, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost



"""
Tensor class using fp16 only. 
- Using struct.pack('e', value) to store values in a list of bytes.
- Always stores the data on the GPU using cudaMalloc. If we need it on the CPU, we can copy it back with cudaMemcpy.
"""

class Tensor:
    def __init__(self, shape, values=None):
        self.shape    = shape
        self.data     = []
        self.size     = self.size()
        self.type     = 'fp16'
        self.gpu_data = None 
        self.cpu_data = None

        self.gpu_data = self._alloc()
       
    
    def GPU(self, values):
        """Copy values from CPU to GPU."""
        # Pack values into fp16 format
        packed_values = b''.join(struct.pack('e', v) for v in values)
        cuda_memset(self.gpu_data, packed_values, self.size, cudaMemcpyHostToDevice)    

    
    def _alloc(self):
        """Allocate memory on the GPU using cudaMalloc."""
        data   = ctypes.c_void_p()
        status = cuda_malloc(ctypes.byref(data), self.size)
        if status != 0:
            raise RuntimeError("Failed to allocate GPU memory")
        return data 

    def size(self):
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    def _to_c_array(self):
        """Convert data to a raw C-style array for CUDA."""
        return (ctypes.c_ubyte * (len(self.data) * 2))(*b"".join(self.data))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, size={self.size}, type={self.type})"

    def __del__(self):
        """Free GPU memory when the tensor is deleted."""
        if self.gpu_data:
            status = cuda_free(self.gpu_data)
            if status != 0:
                raise RuntimeError("Failed to free GPU memory")