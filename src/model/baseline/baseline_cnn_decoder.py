from torch import nn
from torch.nn import Sequential

from src.model.base_model import BaseModel


class BaselineCNNDecoder(BaseModel):
    """
    Simple CNN Decoder
    """

    def __init__(self, n_channels: int):
        """
        Args:
            n_channels (int): number of channels.
        """
        super().__init__()

        self.net = Sequential(
            nn.ConvTranspose1d(
                in_channels=8 * n_channels,
                out_channels=4 * n_channels,
                kernel_size=7,
                stride=5,
                padding=3,
                output_padding=4,
            ),
            nn.ELU(),
            nn.ConvTranspose1d(
                in_channels=4 * n_channels,
                out_channels=2 * n_channels,
                kernel_size=7,
                stride=5,
                padding=3,
                output_padding=4,
            ),
            nn.ELU(),
            nn.ConvTranspose1d(
                in_channels=2 * n_channels,
                out_channels=n_channels,
                kernel_size=7,
                stride=4,
                padding=3,
                output_padding=3,
            ),
            nn.ELU(),
            nn.ConvTranspose1d(
                in_channels=n_channels,
                out_channels=1,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
        )

    def forward(self, z):
        return self.net(z)
