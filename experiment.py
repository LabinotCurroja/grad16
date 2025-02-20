import sys
sys.dont_write_bytecode = True
# temporary disable bytecode generation





from device.device import fp16_support, is_gpu_available, gpu_memory_allocated, gpu_memory_available, gpu_memory_total
from tensor import Tensor, TensorResult
from logging.log import Logger
from grad import QuantumGrad as qgrad

DEVICE = 0 # GPU device ID - set for later in a smarter way. 


def main():
    print("QuantumGrad Example...")

    t1 = qgrad.ones((5, 3))
    t2 = qgrad.ones((3, 1))


    t4 = t1 * t2
    print(t4)

    #t3.backward()
    #print(t3.grads())


    del t1
    del t2
    del t4  

if __name__ == "__main__":
    main()

