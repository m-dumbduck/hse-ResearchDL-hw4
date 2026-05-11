import torch

from src.metrics.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        raw_length: list,
        **kwargs
    ):
        mask = torch.arange(audio.shape[2], device=audio.device)[None, :] < raw_length[
            :, None
        ].to(audio.device)
        mask = mask.unsqueeze(1)
        masked_squared_error = (audio - reconstructed_audio) ** 2 * mask
        return float(masked_squared_error.sum() / mask.sum())
