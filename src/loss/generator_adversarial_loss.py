import torch
from torch import nn


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_for_reconstructed_audio, **batch):
        generator_loss = 0
        for output in discriminator_for_reconstructed_audio.values():
            generator_loss += torch.mean(torch.clamp(1 - output["logits"], min=0))
        generator_loss /= len(discriminator_for_reconstructed_audio)
        return {"generator_loss": generator_loss}
