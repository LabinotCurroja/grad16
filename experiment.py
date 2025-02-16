import sys
sys.dont_write_bytecode = True
# temporary disable bytecode generation





from device.device import fp16_support, is_gpu_available, gpu_memory_allocated
from tensor import Tensor
from logging.log import Logger



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


    tensor = Tensor((2, 2), [1.5, 2.3,
                             3.7, 4.1])


    logger.info(tensor, 'red')

    logger.info(f"GPU memory allocated: {gpu_memory_allocated()}", 'green')

if __name__ == "__main__":
    main()

