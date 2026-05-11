import torch
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        raw_length,
        sample_rate,
        **kwargs
    ):
        stoi = []
        audio = audio.squeeze(1)
        reconstructed_audio = reconstructed_audio.squeeze(1)
        for original, reconstructed, length, fs in zip(
            audio, reconstructed_audio, raw_length, sample_rate
        ):
            stoi.append(
                short_time_objective_intelligibility(
                    preds=reconstructed[:length],
                    target=original[:length],
                    fs=fs.item(),
                )
            )

        return float(torch.mean(torch.stack(stoi)))
