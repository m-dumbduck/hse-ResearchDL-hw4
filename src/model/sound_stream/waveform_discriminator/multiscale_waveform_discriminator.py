from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.waveform_discriminator.waveform_discriminator import (
    WaveformDiscriminator,
)


class MultiscaleWaveformDiscriminator(BaseModel):
    """
    SoundStream MultiscaleWaveformDiscriminator
    """

    def __init__(self, n_channels: int, max_channels: int):
        """
        Args:
            n_channels (int): number of channels.
            max_channels (int): maximum number of channels.
        """
        super().__init__()

        self.waveform_discriminator_1 = WaveformDiscriminator(n_channels, max_channels)
        self.waveform_discriminator_2 = WaveformDiscriminator(n_channels, max_channels)
        self.waveform_discriminator_4 = WaveformDiscriminator(n_channels, max_channels)

        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, audio, **batch):
        audio_downsampled_by_2 = self.pooling(audio)
        audio_downsampled_by_4 = self.pooling(audio_downsampled_by_2)
        return {
            "output_waveform_discriminator_ds_1": self.waveform_discriminator_1(audio),
            "output_waveform_discriminator_ds_2": self.waveform_discriminator_2(
                audio_downsampled_by_2
            ),
            "output_waveform_discriminator_ds_4": self.waveform_discriminator_4(
                audio_downsampled_by_4
            ),
        }
