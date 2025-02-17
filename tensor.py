import struct
import ctypes
from device.device import cuda_malloc, cuda_free, cuda_memset, cuda_memcopy, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, matmul, add, transpose
from cuda.cuda import cudaStatus


"""
  Tensor class using fp16 only.  
  Using struct.pack('e', value) to store values in a list of bytes.

  Always stores the data on the GPU using cudaMalloc. If we need it on the CPU, we can copy it back with cudaMemcpy.
  Basically we dont have GPU() in our framework, as we always store the data on the GPU. 
  What we do have is CPU() to copy the data back to the CPU. - This allows us to show the tensor data in the __repr__ method.
  Other than that, the data should never be manipulated on CPU. 

  To compute the matmul operation for example, we need the accumulated result to be in FP32. This leads to higher precision. 
  What we need to do is to create a TensorResult class that inherits from Tensor. This class will store the data in FP32.
  We then convert the result back to FP16 after the computation is done with a TensorResult to Tensor conversion.

"""






class Tensor:
    def __init__(self, shape, values=None, requires_grad=False, op=None, parents=()):
        self.shape                 = shape
        self.data                  = []
        self.size_in_elements      = self.elements_count()
        self.size_in_bytes         = self.size_in_bytes()
        self.type                  = 'fp16'
        self.gpu_data              = None 
        self.cpu_data              = values

        #DAG
        self.grad                  = None  # Gradient storage in a Tensor
        self.grad_fn               = None
        self.op                    = op 
        self.parents               = parents
        self.requires_grad         = requires_grad


        self.gpu_data = self._alloc()

        if(self.requires_grad):
            self.grad     = self._alloc()

        if values:
            self.GPU(values)

    def CPU(self):
        """Copy data from GPU to CPU and unpack it into a list of Python floats. - Should probably never be used in practice other than debugging."""
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

    def fill(self, value):
        """Fill the tensor with a single value."""
        cpu_data = [value] * self.size_in_elements
        self.GPU(cpu_data)

    def GPU(self, values=None, pointer=None):
        """Copy values from CPU to GPU."""
        # Pack values into fp16 format
        if values:
            self.cpu_data = values

        if pointer is None:
            pointer = self.gpu_data

        packed_values = b''.join(struct.pack('e', v) for v in self.cpu_data)
        cuda_memset(pointer, packed_values, self.size_in_bytes)   

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

    def T(self):
        """Transpose the tensor."""
        T = Tensor((self.shape[1], self.shape[0]), requires_grad=self.requires_grad, op='transpose', parents=(self,))
        transpose(self.gpu_data, T.gpu_data, self.shape[0], self.shape[1])
        return T

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
            
        if self.grad:
            status = cuda_free(self.grad)
            if status != 0:
                raise RuntimeError("Failed to free GPU memory")
            

    def __mul__(self, other):
        """Element-wise multiplication of two tensors."""
        if self.shape != other.shape:
            raise ValueError("Shapes do not match")
        assert isinstance(other, Tensor), "Element-wise multiplication requires a Tensor"

        out = self.matmul(other)
        return out 


    def matmul(self, other):
        #@TODO: Implement batch operations
        """ Matrix multiplication using the kernel. Does not yet support batch operations. """
        assert isinstance(other, Tensor), "Matrix multiplication requires a Tensor"
        assert self.shape[-1] == other.shape[0], "Incompatible shapes for matmul"

        result_shape = (self.shape[0], other.shape[1])
        C = TensorResult(result_shape)

        matmul(self.gpu_data, other.gpu_data, C.gpu_data, self.shape[0], other.shape[1], self.shape[1])

        # we now need to create a tensor result class to store the result in FP16. 
        T = Tensor(result_shape, requires_grad=self.requires_grad or other.requires_grad, op='matmul', parents=(self, other))
        T.GPU(C.CPU())

        # Define the function that computes the gradients
        def matmul_backward(grad):
            # Backpropagate gradients to parents using chain rule
            # For matmul, the gradients are the following:
            # Gradients w.r.t. self and other

            # memory management: 
            t_other    = other.T()
            t_self     = self.T()
            gwr_self   = TensorResult((self.shape[1], self.shape[0]))
            gwr_other  = TensorResult((other.shape[1], other.shape[0]))


            grad_self  = matmul(grad.gpu_data, t_other.gpu_data, gwr_self.gpu_data, self.shape[0], other.shape[1], self.shape[1])
            grad_other = matmul(t_self.gpu_data, grad.gpu_data, gwr_other.gpu_data, self.shape[1], self.shape[0], other.shape[1])

            # memory management 
            del t_other
            del t_self
            del gwr_self
            del gwr_other

            return (grad_self, grad_other)

        # Assign the backward function to the operation
        T.grad_fn = matmul_backward

        del C
        return T



    def compute_grad(self, parent, grad_output):
        """Compute the gradient for a parent tensor."""
        # This will be operation-dependent, and you should define how gradients
        # are computed for each operation type (matmul, addition, etc.).
        if parent.op == "matmul":
            return self.matmul_grad(parent, grad_output)
        # Add logic for other operations like addition, etc.
        return grad_output


    def __add__(self, other):
        """Element-wise addition of two tensors."""
        if self.shape != other.shape:
            raise ValueError("Shapes do not match")
        assert isinstance(other, Tensor), "Element-wise addition requires a Tensor"

        out = self.add(other)
        return out
    
    def add(self, other):
        """ Element-wise addition using the kernel. """
        assert isinstance(other, Tensor), "Element-wise addition requires a Tensor"
        assert self.shape == other.shape, "Incompatible shapes for addition"

        result_shape = self.shape
        C = TensorResult(result_shape)

        add(self.gpu_data, other.gpu_data, C.gpu_data, self.shape[0], self.shape[1], 1)

        T = Tensor(result_shape)
        T.GPU(C.CPU())

        del C
        return T




""" Tensor result class. As we need to accumulate the result of a matmul/* operation(s) in a FP32 to not lose the precision when calculating. Only after that, do we 
    convert the result back to FP16. 
    It is an intermediate value that is used to store the result of a matmul operation, and immediately converted back to FP16 and deleted. 
"""

class TensorResult(Tensor): 
    def __init__(self, shape):
        self.data  = None
        self.shape = shape
        self.type  = 'fp32'
        self.gpu_data = None 
        self.size_in_bytes         = self.size_in_bytes()
        self.size_in_elements      = self.elements_count()

        self.gpu_data = self._alloc()

    def _alloc(self):
        """Allocate memory on the GPU using cudaMalloc, but this time in FP32"""
        data   = ctypes.c_void_p()
        status = cuda_malloc(ctypes.byref(data), self.size_in_bytes)
        if status != 0:
            raise RuntimeError("Failed to allocate GPU memory ", cudaStatus[status])
        return data 
    

    def CPU(self):
        """Copy data from GPU to CPU and unpack it into a list of Python floats (32-bit)."""
        # Allocate a byte array to hold the data copied from the GPU
        cpu_data_bytes = (ctypes.c_uint8 * self.size_in_bytes)()  # Use uint8 for raw bytes
        
        # Copy data from GPU to CPU
        cuda_memcopy(cpu_data_bytes, self.gpu_data, self.size_in_bytes)
        
        # Unpack the byte array into a list of float32 values
        cpu_data = []
        for i in range(0, self.size_in_bytes, 4):  # float32 values are 4 bytes each
            # Extract 4 bytes for each float32 value
            float32_bytes = bytes(cpu_data_bytes[i:i+4])  # Convert to bytes
            # Unpack the bytes into a Python float32 (float)
            float32_value = struct.unpack('f', float32_bytes)[0]
            cpu_data.append(float32_value)
        
        return cpu_data

    def elements_count(self):
        """Return the number of elements in the tensor."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod  
    
    def size_in_bytes(self):
        """Return the size of the tensor in bytes."""
        return self.elements_count() * 4  # fp32 values are 4 bytes each
    
    def __repr__(self):
        """Return a string representation of the tensor."""
        if self.size_in_elements > 256:  # too large to render anyway
            return f"TensorResult(shape={self.shape}, size={self.size_in_elements}, type={self.type}, memory={self.size_in_bytes / 1048576:.2f} MB)"
        
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
                row_values.append(f"{val:.1f}")  # Format value with 4 decimal places for FP32
            rows.append("  " + " ".join(row_values))  # Join values with space for the row
        
        # Format the matrix representation
        matrix_str = "\n".join(rows)
        return f"TensorResult(shape={self.shape}, size={self.size_in_elements}, type={self.type}, memory={self.size_in_bytes} bytes\n[\n{matrix_str}\n])"
    
    def __del__(self):
        print("calling tensor result destructor")
        """Free GPU memory when the tensor is deleted."""
        if self.gpu_data:
            status = cuda_free(self.gpu_data)
            if status != 0:
                raise RuntimeError("Failed to free GPU memory")