import torch
import torchaudio
from torch import nn


class GeneratorMultiscaleReconstructionLoss(nn.Module):
    def __init__(self, sample_rate, window_length_list, n_mels, power):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.register_buffer(
            "alphas",
            torch.tensor(
                [
                    torch.sqrt(torch.tensor(window_length) / 2)
                    for window_length in window_length_list
                ]
            ),
        )
        self.mel_transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    win_length=window_length,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=n_mels,
                    power=power,
                )
                for window_length in window_length_list
            ]
        )

    def forward(
        self,
        audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        raw_length: list,
        **batch
    ):
        if not torch.all(raw_length == audio.shape[-1]):
            losses = []
            for original, reconstructed, length in zip(
                audio, reconstructed_audio, raw_length
            ):
                mel_specs = [
                    transform(original[:, :length]) for transform in self.mel_transforms
                ]
                reconstructed_mel_specs = [
                    transform(reconstructed[:, :length])
                    for transform in self.mel_transforms
                ]
                losses.append(
                    torch.stack(
                        [
                            self.l1_loss(mel_spec, reconstructed_mel_spec)
                            + alpha
                            * self.l2_loss(
                                torch.log(mel_spec + 1e-12),
                                torch.log(reconstructed_mel_spec + 1e-12),
                            )
                            for mel_spec, reconstructed_mel_spec, alpha in zip(
                                mel_specs, reconstructed_mel_specs, self.alphas
                            )
                        ]
                    )
                )
            reconstruction_loss = torch.mean(torch.stack(losses))
            return {"generator_loss": reconstruction_loss}
        mel_specs = [transform(audio) for transform in self.mel_transforms]
        reconstructed_mel_specs = [
            transform(reconstructed_audio) for transform in self.mel_transforms
        ]
        reconstruction_losses = torch.stack(
            [
                self.l1_loss(mel_spec, reconstructed_mel_spec)
                + alpha
                * self.l2_loss(
                    torch.log(mel_spec + 1e-12),
                    torch.log(reconstructed_mel_spec + 1e-12),
                )
                for mel_spec, reconstructed_mel_spec, alpha in zip(
                    mel_specs, reconstructed_mel_specs, self.alphas
                )
            ]
        )
        reconstruction_loss = torch.mean(reconstruction_losses)
        return {"generator_loss": reconstruction_loss}
