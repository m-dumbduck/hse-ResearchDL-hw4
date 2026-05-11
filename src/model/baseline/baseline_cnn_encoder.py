from torch import nn
from torch.nn import Sequential

from src.model.base_model import BaseModel


class BaselineCNNEncoder(BaseModel):
    """
    Simple CNN Encoder
    """

    def __init__(self, n_channels: int):
        """
        Args:
            n_channels (int): number of channels.
        """
        super().__init__()

        self.net = Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=n_channels,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=2 * n_channels,
                kernel_size=7,
                stride=4,
                padding=3,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=2 * n_channels,
                out_channels=4 * n_channels,
                kernel_size=7,
                stride=5,
                padding=3,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=4 * n_channels,
                out_channels=8 * n_channels,
                kernel_size=7,
                stride=5,
                padding=3,
            ),
        )

    def forward(self, audio):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return self.net(audio)
