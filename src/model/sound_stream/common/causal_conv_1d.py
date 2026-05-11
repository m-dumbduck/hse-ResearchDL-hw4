import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class CausalConv1D(BaseModel):
    """
    Causal Convolution 1D
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

        self.pad_size = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, X):
        padded_X = F.pad(X, pad=(self.pad_size, 0), value=0)
        return self.conv(padded_X)
