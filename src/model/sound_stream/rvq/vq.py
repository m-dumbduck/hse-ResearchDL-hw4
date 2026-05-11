import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class VQ(BaseModel):
    """
    SoundStream VQ (one codebook)
    """

    def __init__(self, codebook_size: int, embedding_dim: int, ema_coef: float):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.ema_coef = ema_coef

        self.codebook = nn.Embedding(
            num_embeddings=codebook_size, embedding_dim=embedding_dim
        )
        for param in self.codebook.parameters():
            param.requires_grad = False

        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.weight.detach().clone())
        self.register_buffer("done_initial_clustering", torch.tensor(False))

    def forward(self, X):
        X = X.transpose(1, 2)
        with torch.autocast(device_type=X.device.type, enabled=False):
            X_float32 = X.float()
            closest_indexes = torch.cdist(
                X_float32.detach(), self.codebook.weight
            ).argmin(dim=-1)
            quantized = self.codebook(closest_indexes)
            if self.training:
                self.update_codebook(
                    X_float32.detach().flatten(0, 1), closest_indexes.flatten(0, 1)
                )
        pred = X + (quantized - X).detach()
        return {
            "quantized": pred.transpose(1, 2),
            "quantized_raw": quantized.transpose(1, 2),
            "vq_input": X.transpose(1, 2),
            "vq_indexes": closest_indexes,
        }

    @torch.no_grad()
    def update_codebook(self, X, indexes):
        current_counts = torch.bincount(indexes, minlength=self.codebook_size)
        self.ema_count.mul_(self.ema_coef).add_(current_counts, alpha=1 - self.ema_coef)
        current_embed_sum = torch.zeros_like(self.codebook.weight)
        current_embed_sum.index_add_(0, indexes, X)
        self.ema_embed_sum.mul_(self.ema_coef).add_(
            current_embed_sum, alpha=1 - self.ema_coef
        )
        self.codebook.weight.copy_(
            self.ema_embed_sum / self.ema_count.clamp(min=1e-5).unsqueeze(1)
        )
        dead_mask = self.ema_count < 2
        if dead_mask.any():
            rand_indexes = torch.randint(
                0, X.shape[0], (dead_mask.sum().item(),), device=X.device
            )
            replacement = X[rand_indexes]
            self.codebook.weight[dead_mask] = replacement
            self.ema_embed_sum[dead_mask] = replacement
            self.ema_count[dead_mask] = 3.0

    @torch.no_grad()
    def init_embeddings(self, X, num_iters):
        self.done_initial_clustering.fill_(True)
        X = X.transpose(1, 2).detach()
        input_embeds = X.flatten(0, 1).clone()
        centers_indexes = torch.randperm(
            input_embeds.shape[0], device=input_embeds.device
        )[: self.codebook_size]
        centers = input_embeds[centers_indexes].clone()

        for _ in range(num_iters):
            closest_indexes = torch.cdist(input_embeds, centers).argmin(dim=1)
            new_centers = torch.zeros_like(centers)
            new_centers.index_add_(0, closest_indexes, input_embeds)
            counts = torch.bincount(
                closest_indexes, minlength=self.codebook_size
            ).clamp(min=1)
            new_centers /= counts.unsqueeze(1)
            centers = new_centers

        self.codebook.weight.copy_(centers)
        self.ema_count.fill_(5)
        self.ema_embed_sum.copy_(centers)

        return centers[torch.cdist(X, centers).argmin(dim=2)].transpose(1, 2)
