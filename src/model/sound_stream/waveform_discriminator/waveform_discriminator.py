import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class WaveformDiscriminator(BaseModel):
    """
    SoundStream WaveformDiscriminator
    """

    def __init__(self, n_channels: int, max_channels: int):
        """
        Args:
            n_channels (int): number of channels.
            max_channels (int): maximum number of channels.
        """
        super().__init__()

        groups = n_channels // 4

        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=n_channels,
                    kernel_size=15,
                    stride=1,
                    padding=7,
                ),
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=min(4 * n_channels, max_channels),
                    kernel_size=41,
                    stride=4,
                    groups=groups,
                    padding=20,
                ),
                nn.Conv1d(
                    in_channels=min(4 * n_channels, max_channels),
                    out_channels=min(16 * n_channels, max_channels),
                    kernel_size=41,
                    stride=4,
                    groups=4 * groups,
                    padding=20,
                ),
                nn.Conv1d(
                    in_channels=min(16 * n_channels, max_channels),
                    out_channels=min(64 * n_channels, max_channels),
                    kernel_size=41,
                    stride=4,
                    groups=16 * groups,
                    padding=20,
                ),
                nn.Conv1d(
                    in_channels=min(64 * n_channels, max_channels),
                    out_channels=min(256 * n_channels, max_channels),
                    kernel_size=41,
                    stride=4,
                    groups=64 * groups,
                    padding=20,
                ),
                nn.Conv1d(
                    in_channels=min(256 * n_channels, max_channels),
                    out_channels=min(256 * n_channels, max_channels),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
                nn.Conv1d(
                    in_channels=min(256 * n_channels, max_channels),
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]
        )

    def forward(self, audio, **batch):
        X = audio
        features = {}
        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i != len(self.layers) - 1:
                features[f"layer_{i}"] = X
                X = F.leaky_relu(X, negative_slope=0.2)
        return {"logits": X, "features": features}
