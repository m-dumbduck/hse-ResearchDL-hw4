import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel
from src.model.sound_stream.rvq.vq import VQ


class RVQ(BaseModel):
    """
    SoundStream RVQ
    """

    def __init__(
        self, n_codebooks: int, codebook_size: int, embedding_dim: int, ema_coef: float
    ):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.ema_coef = ema_coef
        self.register_buffer("done_initial_clustering", torch.tensor(False))

        self.vqs = nn.ModuleList(
            [
                VQ(
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    ema_coef=ema_coef,
                )
                for _ in range(n_codebooks)
            ]
        )

    def forward(self, X):
        residual = X.detach()
        quantized = 0
        indexes = torch.zeros(
            size=(X.shape[0], self.n_codebooks, X.shape[2]),
            device=X.device,
            dtype=torch.long,
        )
        for i_vq, vq in enumerate(self.vqs):
            vq_output = vq(residual)
            indexes[:, i_vq, :] = vq_output["vq_indexes"]
            quantized = quantized + vq_output["quantized_raw"]
            residual = residual - vq_output["quantized_raw"]
        pred = X + (quantized - X).detach()
        return {
            "quantized": pred,
            "quantized_raw": quantized,
            "rvq_input": X,
            "rvq_indexes": indexes,
        }

    def decode_from_indexes(self, indexes):
        result = 0
        for i, vq in enumerate(self.vqs):
            result += vq.decode_from_indexes(indexes[:, i, :])
        return result

    @torch.no_grad()
    def init_embeddings(self, X, num_iters):
        self.done_initial_clustering.fill_(True)
        X = X.detach()
        quantized = 0
        for vq in self.vqs:
            current_quantized = vq.init_embeddings(X, num_iters).detach()
            quantized = quantized + current_quantized
            X = X - current_quantized
        return quantized
