
"""
    This is the module that contains the parent for the system. It is used to initalize Tensors, and to perform operations on them.
    Basically this class is what users use only. 
"""

from .tensor import Tensor



class QuantumGrad:
    def __init__(self):
        # we check here for the device, and if it is available.
        pass

    def zeros(self, shape):
        """Create a tensor filled with zeros."""
        return Tensor(shape)
    
    def ones(self, shape):
        """Create a tensor filled with ones."""
        t = Tensor(shape)
        t.fill(1)
        return t
    
    def randn(self, shape):
        """Create a tensor filled with random numbers."""
        t = Tensor(shape)
        return t 

    def Tensor(self, shape):
        """Create a tensor."""
        return Tensor(shape)
    
    def matmul(self, a, b):
        """Matrix multiplication."""
        return a.matmul(b)

