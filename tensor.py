import struct
import ctypes



"""
Tensor class using fp16 only. 
- Using struct.pack('e', value) to store values in a list of bytes.
"""

class Tensor:
    def __init__(self, shape, values=None):
        self.shape = shape
        self.data  = []
        self.type  = 'fp16'
        
        # store values with cudaMalloc - very important for performance
        # either stores values or initializes with zeros
        if values:
            if len(values) != self.length():
                raise ValueError("Number of values does not match tensor size.")
            self.data = [struct.pack('e', v) for v in values]
        else:
            self.data = [struct.pack('e', 0.0)] * self.length()


    def length(self):
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    def GPU(self):
        return
    
    def CPU(self):
        return

    def _to_c_array(self):
        """Convert data to a raw C-style array for CUDA."""
        return (ctypes.c_ubyte * (len(self.data) * 2))(*b"".join(self.data))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, length={self.length()}, type={self.type})"



# Example Usage
tensor = Tensor((2, 2), [1.5, 2.3, 3.7, 4.1])
print(tensor)
