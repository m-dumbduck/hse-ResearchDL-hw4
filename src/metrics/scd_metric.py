import librosa

from src.metrics.base_metric import BaseMetric


class SpectralCentroidMetric(BaseMetric):
    def __init__(self, use_reconstructed: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reconstructed = use_reconstructed

    def __call__(self, audio, reconstructed_audio, raw_length, sample_rate, **kwargs):
        values = []
        source = reconstructed_audio if self.use_reconstructed else audio
        source = source.squeeze(1)
        for waveform, length, sr in zip(source, raw_length, sample_rate):
            waveform = waveform[: length.item()].detach().cpu().numpy()
            centroid = librosa.feature.spectral_centroid(
                y=waveform, sr=sr.item()
            ).mean()
            values.append(centroid)
        return float(sum(values) / len(values))
