import torch
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment

from src.metrics.base_metric import BaseMetric


class NISQAMOSMetric(BaseMetric):
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
        mos_scores = []
        audio = audio.squeeze(1)
        reconstructed_audio = reconstructed_audio.squeeze(1)
        for original, reconstructed, length, fs in zip(
            audio, reconstructed_audio, raw_length, sample_rate
        ):
            mos_scores.append(
                non_intrusive_speech_quality_assessment(
                    preds=reconstructed[:length].detach(),
                    fs=fs.item(),
                )[0]
            )

        return float(torch.mean(torch.stack(mos_scores)))
