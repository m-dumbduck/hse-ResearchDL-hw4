from torch import nn
from torch.nn import Sequential

from src.model.base_model import BaseModel
from src.model.sound_stream.common import CausalConv1D
from src.model.sound_stream.encoder.encoder_block import EncoderBlock


class Encoder(BaseModel):
    """
    SoundStream Encoder
    """

    def __init__(self, n_channels: int, latent_channels: int):
        """
        Args:
            n_channels (int): number of channels.
            latent_channels (int): number of channels for latent output.
        """
        super().__init__()

        self.net = Sequential(
            CausalConv1D(in_channels=1, out_channels=n_channels, kernel_size=7),
            EncoderBlock(n_channels=2 * n_channels, stride=2),
            EncoderBlock(n_channels=4 * n_channels, stride=4),
            EncoderBlock(n_channels=8 * n_channels, stride=5),
            EncoderBlock(n_channels=16 * n_channels, stride=5),
            CausalConv1D(
                in_channels=16 * n_channels, out_channels=latent_channels, kernel_size=3
            ),
        )

    def forward(self, X):
        return self.net(X)
