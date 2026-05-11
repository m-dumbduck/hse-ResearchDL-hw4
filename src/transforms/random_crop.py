import torch
from torch import nn


class RandomCrop1D(nn.Module):
    def __init__(self, data_object_key: str, length: int):
        super().__init__()
        self.data_object_key = data_object_key
        self.length = length

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): cropped tensor.
        """
        start = torch.randint(0, x[self.data_object_key].shape[-1] - self.length, (1,))
        x[self.data_object_key] = x[self.data_object_key][
            ..., start : start + self.length
        ]
        return x
