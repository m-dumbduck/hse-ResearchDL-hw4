import torch
import torchaudio
from torch import nn


class ResampleAudio(nn.Module):
    def __init__(
        self, data_object_key: str, sample_rate_key: int, target_sample_rate: int
    ):
        super().__init__()
        self.data_object_key = data_object_key
        self.sample_rate_key = sample_rate_key
        self.target_sample_rate = target_sample_rate

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): padded tensor.
        """
        if x[self.sample_rate_key] == self.target_sample_rate:
            return x
        x[self.data_object_key] = torchaudio.functional.resample(
            x[self.data_object_key], x[self.sample_rate_key], self.target_sample_rate
        )
        return x
