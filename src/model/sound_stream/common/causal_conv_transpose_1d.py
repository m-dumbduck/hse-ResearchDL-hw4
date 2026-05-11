import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class CausalConvTranspose1D(BaseModel):
    """
    Causal Convolution Transpose 1D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.crop_size = kernel_size - stride
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, X):
        X = self.conv_transpose(X)
        if self.crop_size > 0:
            X = X[..., : -self.crop_size]
        return X
