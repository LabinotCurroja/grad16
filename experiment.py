import sys
sys.dont_write_bytecode = True
# temporary disable bytecode generation





from device.device import fp16_support, is_gpu_available
from tensor import Tensor
from logging.log import Logger



def main():
    logger = Logger("Experiment", debug=True)

    if not is_gpu_available():
        logger.FATAL("No CUDA device available.")
        return
    
    logger.info("CUDA device is available.")


    if not fp16_support():
        logger.FATAL("CUDA device does not support fp16.")
        return
    
    logger.info("CUDA device supports fp16.")

    tensor = Tensor((2, 2), [1.5, 2.3, 3.7, 4.1])


    logger.warn(tensor)


if __name__ == "__main__":
    main()

