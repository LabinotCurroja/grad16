import struct
import ctypes
from device.device import cuda_malloc, cuda_free, cuda_memset, cuda_memcopy, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost



"""
Tensor class using fp16 only. 
- Using struct.pack('e', value) to store values in a list of bytes.
- Always stores the data on the GPU using cudaMalloc. If we need it on the CPU, we can copy it back with cudaMemcpy.
  Basically we dont have GPU() in our framework, as we always store the data on the GPU. 
  What we do have is CPU() to copy the data back to the CPU.
"""

class Tensor:
    def __init__(self, shape, values=None):
        self.shape    = shape
        self.data     = []
        self.size     = self.size()
        self.type     = 'fp16'
        self.gpu_data = None 
        self.cpu_data = values


        self.gpu_data = self._alloc()

        if values:
            self.GPU(values)

    def CPU(self):
        """Copy data from GPU to CPU and unpack it into a list of Python floats."""
        # Allocate a byte array to hold the data copied from the GPU
        cpu_data_bytes = (ctypes.c_uint8 * self.size )()  # Use uint8 for raw bytes
        
        # Copy data from GPU to CPU
        cuda_memcopy(cpu_data_bytes, self.gpu_data, self.size)
        
        # Unpack the byte array into a list of fp16 values
        cpu_data = []
        for i in range(0, self.size, 2):  # fp16 values are 2 bytes each
            # Extract 2 bytes for each fp16 value
            fp16_bytes = bytes(cpu_data_bytes[i:i+2])  # Convert to bytes
            # Unpack the bytes into a Python float
            fp16_value = struct.unpack('e', fp16_bytes)[0]
            cpu_data.append(fp16_value)
        
        return cpu_data
        
    
    def GPU(self, values=None):
        """Copy values from CPU to GPU."""
        # Pack values into fp16 format
        if values:
            self.cpu_data = values

        packed_values = b''.join(struct.pack('e', v) for v in self.cpu_data)
        cuda_memset(self.gpu_data, packed_values, self.size)   

        # always clear CPU data. We dont want to use memory there unless absolutely needed. 
        self.cpu_data = None
    
    def _alloc(self):
        """Allocate memory on the GPU using cudaMalloc."""
        data   = ctypes.c_void_p()
        status = cuda_malloc(ctypes.byref(data), self.size)
        if status != 0:
            raise RuntimeError("Failed to allocate GPU memory")
        return data 

    def size(self):
        """Return the number of elements in the tensor."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod * 2

    def _to_c_array(self):
        """Convert data to a raw C-style array for CUDA."""
        return (ctypes.c_ubyte * (len(self.data) * 2))(*b"".join(self.data))

    def __repr__(self):
        """Return a string representation of the tensor."""
        if self.size > 32: # too large to render anyways. 
            return f"Tensor(shape={self.shape}, size={self.size}, type={self.type})"
        
        cdata         = self.CPU()
        tensor_values = cdata  
        rows          = []
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                val = tensor_values[i * self.shape[1] + j]
                rows.append(f"  {val:.1f}")
        
        # Format the matrix representation
        matrix_str = "\n".join(rows)
        return f"Tensor shape={self.shape}, size={self.size}, type={self.type}, values=\n[\n{matrix_str}\n]"


    def __del__(self):
        """Free GPU memory when the tensor is deleted."""
        if self.gpu_data:
            status = cuda_free(self.gpu_data)
            if status != 0:
                raise RuntimeError("Failed to free GPU memory")