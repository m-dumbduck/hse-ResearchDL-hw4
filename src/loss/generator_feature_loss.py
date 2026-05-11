import torch
from torch import nn


class GeneratorFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, discriminator_for_audio, discriminator_for_reconstructed_audio, **batch
    ):
        generator_loss = 0
        total_features = 0
        for key in discriminator_for_audio.keys():
            for feature_for_audio, feature_for_reconstructed_audio in zip(
                discriminator_for_audio[key]["features"].values(),
                discriminator_for_reconstructed_audio[key]["features"].values(),
            ):
                generator_loss += torch.mean(
                    torch.abs(feature_for_audio - feature_for_reconstructed_audio)
                )
                total_features += 1
        generator_loss /= total_features
        return {"generator_loss": generator_loss}
