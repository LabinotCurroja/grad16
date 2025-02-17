import sys
sys.dont_write_bytecode = True
# temporary disable bytecode generation





from device.device import fp16_support, is_gpu_available, gpu_memory_allocated, gpu_memory_available, gpu_memory_total
from tensor import Tensor, TensorResult
from logging.log import Logger


DEVICE = 0 # GPU device ID - set for later in a smarter way. 


def main():
    logger = Logger("Grad16", debug=True)

    if not is_gpu_available():
        logger.FATAL("No CUDA device available.")
        return
    
    logger.info("CUDA device is available.")

    if not fp16_support():
        logger.FATAL("CUDA device does not support fp16.")
        return
    
    logger.info("CUDA device supports fp16.")
    logger.info("Using GPU device: [0]")

    # now we just need to fill the tensors.
    tx = TensorResult((16, 16))

    t1 = Tensor((16, 16))
    t2 = Tensor((16, 16))

    t1.fill(1)
    t2.fill(2)

    t3 = t1.matmul(t2)

    #print(t1)
    #print(t2)
    #print(t3)

    del t1 
    del t2
    del t3
    del tx
    #logger.info(tensor, 'red')
    logger.info(f"GPU memory allocated : {gpu_memory_allocated()} MB")
    logger.info(f"GPU memory available : {gpu_memory_available()} MB")
    logger.info(f"GPU memory total     : {gpu_memory_total()} MB")

if __name__ == "__main__":
    main()

