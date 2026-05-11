import torch
from torch import nn

from src.loss.generator_adversarial_loss import GeneratorAdversarialLoss
from src.loss.generator_commitment_loss import GeneratorCommitmentLoss
from src.loss.generator_feature_loss import GeneratorFeatureLoss
from src.loss.generator_multiscale_reconstruction_loss import (
    GeneratorMultiscaleReconstructionLoss,
)


class SoundStreamGeneratorLoss(nn.Module):
    def __init__(
        self,
        adv_loss: GeneratorAdversarialLoss,
        adv_loss_weight: float,
        feature_loss: GeneratorFeatureLoss,
        feature_loss_weight: float,
        multiscale_reconstruction_loss: GeneratorMultiscaleReconstructionLoss,
        multiscale_reconstruction_loss_weight: float,
        commitment_loss: GeneratorCommitmentLoss,
        commitment_loss_weight: float,
    ):
        super().__init__()
        self.adv_loss = adv_loss
        self.adv_loss_weight = adv_loss_weight
        self.feature_loss = feature_loss
        self.feature_loss_weight = feature_loss_weight
        self.multiscale_reconstruction_loss = multiscale_reconstruction_loss
        self.multiscale_reconstruction_loss_weight = (
            multiscale_reconstruction_loss_weight
        )
        self.commitment_loss = commitment_loss
        self.commitment_loss_weight = commitment_loss_weight

    def forward(self, **batch):
        generator_adv_loss = self.adv_loss(**batch)["generator_loss"]
        generator_feature_loss = self.feature_loss(**batch)["generator_loss"]
        generator_multiscale_reconstruction_loss = self.multiscale_reconstruction_loss(
            **batch
        )["generator_loss"]
        commitment_loss = self.commitment_loss(**batch)["generator_loss"]
        generator_loss = (
            self.adv_loss_weight * generator_adv_loss
            + self.feature_loss_weight * generator_feature_loss
            + self.multiscale_reconstruction_loss_weight
            * generator_multiscale_reconstruction_loss
            + self.commitment_loss_weight * commitment_loss
        )
        return {
            "generator_loss": generator_loss,
            "generator_adv_loss": generator_adv_loss,
            "generator_feature_loss": generator_feature_loss,
            "generator_multiscale_reconstruction_loss": generator_multiscale_reconstruction_loss,
            "generator_commitment_loss": commitment_loss,
        }
