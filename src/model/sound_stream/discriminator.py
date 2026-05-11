from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.stft_discriminator import STFTDiscriminator
from src.model.sound_stream.waveform_discriminator import (
    MultiscaleWaveformDiscriminator,
)


class Discriminator(BaseModel):
    """
    SoundStream Discriminator (consisting of STFTDiscriminator and MultiscaleWaveformDiscriminator).
    """

    def __init__(
        self,
        multiscale_waveform_discriminator: MultiscaleWaveformDiscriminator,
        stft_discriminator: STFTDiscriminator,
    ):
        super().__init__()

        self.multiscale_waveform_discriminator = multiscale_waveform_discriminator
        self.stft_discriminator = stft_discriminator

    def forward(self, audio, **batch):
        multiscale_waveform_discriminator_outputs = (
            self.multiscale_waveform_discriminator(audio)
        )
        stft_discriminator_outputs = self.stft_discriminator(audio)
        discriminator_outputs = {}
        discriminator_outputs.update(multiscale_waveform_discriminator_outputs)
        discriminator_outputs["output_stft_discriminator"] = stft_discriminator_outputs
        return discriminator_outputs

    def forward_for_both(self, audio, reconstructed_audio, **batch):
        return {
            "discriminator_for_audio": self.forward(audio),
            "discriminator_for_reconstructed_audio": self.forward(reconstructed_audio),
        }
