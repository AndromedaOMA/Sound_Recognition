from pydantic import BaseModel


class ModelConfigs(BaseModel):
    """conv_block"""
    # convolution_layers
    in_channels: int = 1
    mid_channels: int = 128
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

    # max_pooling
    p_kernel1: int = 4
    p_stride1: int = 4
    p_kernel2: int = 2
    p_stride2: int = 2
