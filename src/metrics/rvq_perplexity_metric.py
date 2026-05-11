import torch

from src.metrics.base_metric import BaseMetric


class RVQPerplexityMetric(BaseMetric):
    def __init__(self, codebook_size, normalized=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_size = codebook_size
        self.normalized = normalized

    def __call__(self, rvq_indexes, **batch):
        values = []
        for i in range(rvq_indexes.shape[1]):
            values.append(self._perplexity(rvq_indexes[:, i, :]))
        value = torch.mean(torch.stack(values))
        if self.normalized:
            value = value / self.codebook_size
        return float(value)

    def _perplexity(self, indexes):
        counts = torch.bincount(indexes.reshape(-1), minlength=self.codebook_size)
        probs = counts / counts.sum().clamp(min=1.0)
        probs = probs[probs > 0]
        return torch.exp(-(probs * torch.log(probs)).sum())
