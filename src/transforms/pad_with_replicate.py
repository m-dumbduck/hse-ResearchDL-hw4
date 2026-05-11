import torch
from torch import nn


class PadWithReplicate(nn.Module):
    def __init__(self, data_object_key: str, length: int):
        super().__init__()
        self.data_object_key = data_object_key
        self.length = length

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): padded tensor.
        """
        x[self.data_object_key] = torch.nn.functional.pad(
            x[self.data_object_key], (0, self.length), mode="replicate"
        )
        return x
