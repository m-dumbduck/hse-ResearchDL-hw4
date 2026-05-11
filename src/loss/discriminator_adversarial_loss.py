import torch
from torch import nn


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, discriminator_for_audio, discriminator_for_reconstructed_audio, **batch
    ):
        loss_1 = 0
        for output in discriminator_for_audio.values():
            loss_1 += torch.mean(torch.clamp(1 - output["logits"], min=0))
        loss_1 /= len(discriminator_for_audio)
        loss_2 = 0
        for output in discriminator_for_reconstructed_audio.values():
            loss_2 += torch.mean(torch.clamp(1 + output["logits"], min=0))
        loss_2 /= len(discriminator_for_reconstructed_audio)
        discriminator_loss = loss_1 + loss_2
        return {"discriminator_loss": discriminator_loss}
