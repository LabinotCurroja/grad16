import struct
import ctypes
from device.device import cuda_malloc, cuda_free, cuda_memset, cuda_memcopy, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost
from cuda.cuda import cudaStatus


"""
Tensor class using fp16 only. 
- Using struct.pack('e', value) to store values in a list of bytes.
- Always stores the data on the GPU using cudaMalloc. If we need it on the CPU, we can copy it back with cudaMemcpy.
  Basically we dont have GPU() in our framework, as we always store the data on the GPU. 
  What we do have is CPU() to copy the data back to the CPU.
"""

class Tensor:
    def __init__(self, shape, values=None):
        self.shape                 = shape
        self.data                  = []
        self.size_in_elements      = self.elements_count()
        self.size_in_bytes         = self.size_in_bytes()
        self.type                  = 'fp16'
        self.gpu_data              = None 
        self.cpu_data              = values


        self.gpu_data = self._alloc()

        if values:
            self.GPU(values)

    def CPU(self):
        """Copy data from GPU to CPU and unpack it into a list of Python floats."""
        # Allocate a byte array to hold the data copied from the GPU
        cpu_data_bytes = (ctypes.c_uint8 * self.size_in_bytes )()  # Use uint8 for raw bytes
        
        # Copy data from GPU to CPU
        cuda_memcopy(cpu_data_bytes, self.gpu_data, self.size_in_bytes)
        
        # Unpack the byte array into a list of fp16 values
        cpu_data = []
        for i in range(0, self.size_in_bytes, 2):  # fp16 values are 2 bytes each
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
        cuda_memset(self.gpu_data, packed_values, self.size_in_bytes)   

        # always clear CPU data. We dont want to use memory there unless absolutely needed. 
        self.cpu_data = None
    
    def _alloc(self):
        """Allocate memory on the GPU using cudaMalloc."""
        data   = ctypes.c_void_p()
        status = cuda_malloc(ctypes.byref(data), self.size_in_bytes)
        if status != 0:
            raise RuntimeError("Failed to allocate GPU memory ", cudaStatus[status])
        return data 

    def elements_count(self):
        """Return the number of elements in the tensor."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod  # fp16 values are 2 bytes each

    def size_in_bytes(self):
        """Return the size of the tensor in bytes."""
        return self.elements_count() * 2  # fp16 values are 2 bytes each



    def __repr__(self):
        """Return a string representation of the tensor."""
        if self.size_in_elements > 256:  # too large to render anyways
            return f"Tensor(shape={self.shape}, size={self.size_in_elements}, type={self.type}, memory={self.size_in_bytes / 1048576:.2f} MB)"
        
        # Get tensor data from the CPU (flattened array)
        cdata = self.CPU()  # Assuming this returns a flat array-like structure
        tensor_values = cdata
        rows = []

        # Loop over rows
        for i in range(self.shape[0]):
            row_values = []  # Collect values for each row
            for j in range(self.shape[1]):
                # Access the tensor element using the flattened index calculation
                val = tensor_values[i * self.shape[1] + j]  
                row_values.append(f"{val:.1f}")  # Format value with one decimal
            rows.append("  " + " ".join(row_values))  # Join values with space for the row
        
        # Format the matrix representation
        matrix_str = "\n".join(rows)
        return f"Tensor(shape={self.shape}, size={self.size_in_elements}, type={self.type}, memory={self.size_in_bytes} bytes\n[\n{matrix_str}\n]"



    def __del__(self):
        """Free GPU memory when the tensor is deleted."""
        if self.gpu_data:
            status = cuda_free(self.gpu_data)
            if status != 0:
                raise RuntimeError("Failed to free GPU memory")
            

    def __mul__(self, other):
        """Element-wise multiplication of two tensors."""
        if self.shape != other.shape:
            raise ValueError("Shapes do not match")
        
        # Get tensor data from the CPU (flattened array)
        cdata1 = self.CPU() 
        cdata2 = other.CPU()
        result = [a * b for a, b in zip(cdata1, cdata2)]


    def __matmul__(self, other):
        """ Matrix multiplication using the kernel. """
        assert isinstance(other, Tensor), "Matrix multiplication requires a Tensor"
        assert self.shape[-1] == other.shape[0], "Incompatible shapes for matmul"

        result_shape = (self.shape[0], other.shape[1])
        #result_gpu = torch.matmul(self.gpu_data, other.gpu_data)

        return Tensor(result_shape, values=result_gpu)