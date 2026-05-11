from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.common import CausalConvTranspose1D, ResidualUnit1D


class DecoderBlock(BaseModel):
    """
    DecoderBlock for decoder
    """

    def __init__(self, n_channels: int, stride: int):
        super().__init__()

        self.net = nn.Sequential(
            CausalConvTranspose1D(
                in_channels=n_channels,
                out_channels=n_channels // 2,
                kernel_size=2 * stride,
                stride=stride,
            ),
            nn.ELU(),
            ResidualUnit1D(n_channels=n_channels // 2, dilation=1),
            nn.ELU(),
            ResidualUnit1D(n_channels=n_channels // 2, dilation=3),
            nn.ELU(),
            ResidualUnit1D(n_channels=n_channels // 2, dilation=9),
        )

    def forward(self, X):
        return self.net(X)
