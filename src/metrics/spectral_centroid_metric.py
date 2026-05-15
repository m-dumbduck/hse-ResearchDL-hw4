import torch

from src.metrics.base_metric import BaseMetric


class SpectralCentroidMetric(BaseMetric):
    def __init__(
        self, use_reconstructed, n_fft, win_length, hop_length, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_reconstructed = use_reconstructed
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def _spectral_centroid(self, wave, sample_rate):
        length = wave.shape[0]

        n_fft = min(self.n_fft, length)
        win_length = min(self.win_length, n_fft)
        hop_length = min(self.hop_length, max(1, length // 4))

        spec = torch.stft(
            wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True,
        ).abs()

        freqs = torch.linspace(
            0,
            sample_rate / 2,
            spec.shape[0],
            device=wave.device,
            dtype=wave.dtype,
        )

        centroid_per_frame = (spec * freqs[:, None]).sum(dim=0) / spec.sum(
            dim=0
        ).clamp_min(1e-12)
        return centroid_per_frame.mean()

    def __call__(self, audio, reconstructed_audio, raw_length, sample_rate, **kwargs):
        values = []
        source = reconstructed_audio if self.use_reconstructed else audio
        source = source.squeeze(1)
        for wave, length, sr in zip(source, raw_length, sample_rate):
            wave = wave[: length.item()]
            centroid = self._spectral_centroid(wave, sr.item())
            values.append(centroid)

        return float(torch.mean(torch.stack(values)))
