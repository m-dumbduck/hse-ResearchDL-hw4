import torch
from torch import nn


class ToTensor(nn.Module):
    def __init__(self, data_object_key: str):
        super().__init__()
        self.data_object_key = data_object_key

    def forward(self, x):
        """
        Args:
            x (Tensor): input array.
        Returns:
            x (Tensor): tensor.
        """
        x[self.data_object_key] = torch.as_tensor(x[self.data_object_key])
        return x
