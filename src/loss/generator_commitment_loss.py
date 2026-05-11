import torch
from torch import nn


class GeneratorCommitmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, rvq_input, quantized_raw, **batch):
        return {"generator_loss": self.mse_loss(rvq_input, quantized_raw)}
