import librosa

from src.metrics.base_metric import BaseMetric


class ZeroCrossingRateMetric(BaseMetric):
    def __init__(self, use_reconstructed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reconstructed = use_reconstructed

    def __call__(self, audio, reconstructed_audio, raw_length, **kwargs):
        values = []
        source = reconstructed_audio if self.use_reconstructed else audio
        source = source.squeeze(1)
        for wave, length in zip(source, raw_length):
            wave = wave[: length.item()].detach().cpu().numpy()
            zcr = librosa.feature.zero_crossing_rate(wave).mean()
            values.append(zcr)
        return float(sum(values) / len(values))
