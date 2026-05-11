import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.common import ResidualUnit2D


class STFTDiscriminator(BaseModel):
    """
    SoundStream STFTDiscriminator
    """

    def __init__(self, n_channels: int, stft_win_length: int, stft_hop_length: int):
        """
        Args:
            n_channels (int): number of channels.
            stft_win_length (int): stft window length.
            stft_hop_length (int): stft hop length.
        """
        super().__init__()

        self.stft_win_length = stft_win_length
        self.stft_hop_length = stft_hop_length

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=2, out_channels=n_channels, kernel_size=7, padding=3
                ),
                ResidualUnit2D(
                    in_channels=n_channels,
                    out_channels=2 * n_channels,
                    stride_t=1,
                    stride_f=2,
                ),
                ResidualUnit2D(
                    in_channels=2 * n_channels,
                    out_channels=4 * n_channels,
                    stride_t=2,
                    stride_f=2,
                ),
                ResidualUnit2D(
                    in_channels=4 * n_channels,
                    out_channels=4 * n_channels,
                    stride_t=1,
                    stride_f=2,
                ),
                ResidualUnit2D(
                    in_channels=4 * n_channels,
                    out_channels=8 * n_channels,
                    stride_t=2,
                    stride_f=2,
                ),
                ResidualUnit2D(
                    in_channels=8 * n_channels,
                    out_channels=8 * n_channels,
                    stride_t=1,
                    stride_f=2,
                ),
                ResidualUnit2D(
                    in_channels=8 * n_channels,
                    out_channels=16 * n_channels,
                    stride_t=2,
                    stride_f=2,
                ),
                nn.Conv2d(
                    in_channels=16 * n_channels,
                    out_channels=1,
                    kernel_size=(1, self.stft_win_length // 128),
                ),
            ]
        )

    def forward(self, audio, **batch):
        stft = self._wave_to_stft(audio)
        X = stft
        features = {}
        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i != len(self.layers) - 1:
                features[f"layer_{i}"] = X
                X = F.leaky_relu(X, negative_slope=0.2)
        return {"logits": X.squeeze(3), "features": features}

    def _wave_to_stft(self, audio):
        audio = audio.squeeze(1)
        stft = torch.stft(
            audio,
            n_fft=self.stft_win_length,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            return_complex=False,
        ).permute(0, 3, 2, 1)
        stft = stft[:, :, :, : self.stft_win_length // 2]
        return stft
