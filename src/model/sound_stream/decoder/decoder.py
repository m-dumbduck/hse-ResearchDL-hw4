from torch import nn
from torch.nn import Sequential

from src.model.base_model import BaseModel
from src.model.sound_stream.common import CausalConv1D
from src.model.sound_stream.decoder.decoder_block import DecoderBlock


class Decoder(BaseModel):
    """
    SoundStream Decoder
    """

    def __init__(self, n_channels: int, latent_channels: int):
        """
        Args:
            n_channels (int): number of channels.
            latent_channels (int): number of channels in latent input.
        """
        super().__init__()

        self.net = Sequential(
            CausalConv1D(
                in_channels=latent_channels, out_channels=16 * n_channels, kernel_size=3
            ),
            DecoderBlock(n_channels=16 * n_channels, stride=5),
            DecoderBlock(n_channels=8 * n_channels, stride=5),
            DecoderBlock(n_channels=4 * n_channels, stride=4),
            DecoderBlock(n_channels=2 * n_channels, stride=2),
            CausalConv1D(in_channels=n_channels, out_channels=1, kernel_size=7),
        )

    def forward(self, X):
        return self.net(X)
