from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.common.causal_conv_1d import CausalConv1D


class ResidualUnit1D(BaseModel):
    """
    Residual Unit 1D for encoder and decoder
    """

    def __init__(self, n_channels: int, dilation: int):
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1D(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=7,
                stride=1,
                dilation=dilation,
            ),
            nn.ELU(),
            CausalConv1D(
                in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1
            ),
        )

    def forward(self, X):
        return self.net(X) + X
