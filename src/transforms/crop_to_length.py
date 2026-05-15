from torch import nn


class CropToLength(nn.Module):
    def __init__(self, data_object_key: str, length: int):
        super().__init__()
        self.data_object_key = data_object_key
        self.length = length

    def forward(self, x):
        x[self.data_object_key] = x[self.data_object_key][..., : self.length]
        return x
