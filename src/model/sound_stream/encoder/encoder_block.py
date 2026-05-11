from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.common import CausalConv1D, ResidualUnit1D


class EncoderBlock(BaseModel):
    """
    EncoderBlock for encoder
    """

    def __init__(self, n_channels: int, stride: int):
        super().__init__()

        self.net = nn.Sequential(
            ResidualUnit1D(n_channels=n_channels // 2, dilation=1),
            nn.ELU(),
            ResidualUnit1D(n_channels=n_channels // 2, dilation=3),
            nn.ELU(),
            ResidualUnit1D(n_channels=n_channels // 2, dilation=9),
            nn.ELU(),
            CausalConv1D(
                in_channels=n_channels // 2,
                out_channels=n_channels,
                kernel_size=2 * stride,
                stride=stride,
            ),
        )

    def forward(self, X):
        return self.net(X)
