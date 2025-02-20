
"""
    This is the module that contains the parent for the system. It is used to initalize Tensors, and to perform operations on them.
    Basically this class is what users use only. 
"""

from tensor import Tensor
from device.device import fp16_support, is_gpu_available, gpu_memory_allocated, gpu_memory_available, gpu_memory_total
import random



class QuantumGrad:
    def zeros(shape):
        """Create a tensor filled with zeros."""
        return Tensor(shape)
    
    def ones(shape):
        """Create a tensor filled with ones."""
        t = Tensor(shape)
        t.fill(1)
        return t
    
    def randf(shape):
        """Create a tensor filled with random numbers."""
        t = Tensor(shape)
        r = [random.gauss(0, 1) for _ in range(shape[1]) for _ in range(shape[0])]
        t.GPU(r)
        del r
        return t 

    def matmul(a, b):
        """Matrix multiplication."""
        return a.matmul(b)

