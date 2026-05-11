from torch import nn

from src.model.base_model import BaseModel
from src.model.baseline.baseline_cnn_decoder import BaselineCNNDecoder
from src.model.baseline.baseline_cnn_encoder import BaselineCNNEncoder


class BaselineGenerator(BaseModel):
    """
    Simple generator
    """

    def __init__(self, n_channels: int):
        """
        Args:
            n_channels (int): number of channels.
        """
        super().__init__()

        self.encoder = BaselineCNNEncoder(n_channels)
        self.decoder = BaselineCNNDecoder(n_channels)

    def forward(self, audio, **batch):
        z = self.encoder(audio)
        reconstructed_audio = self.decoder(z)
        return {"reconstructed_audio": reconstructed_audio}
