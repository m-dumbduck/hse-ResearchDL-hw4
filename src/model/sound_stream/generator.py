from huggingface_hub import PyTorchModelHubMixin

from src.model.base_model import BaseModel
from src.model.sound_stream.decoder import Decoder
from src.model.sound_stream.encoder import Encoder
from src.model.sound_stream.rvq import RVQ


class Generator(BaseModel, PyTorchModelHubMixin):
    """
    Simple generator
    """

    def __init__(
        self,
        n_channels: int,
        latent_channels: int,
        n_codebooks: int,
        codebook_size: int,
        ema_coef: float,
    ):
        """
        Args:
            n_channels (int): number of channels.
            latent_channels (int): number of latent channels.
            n_codebooks (int): number of codebooks for rvq.
            codebook_size (int): number of codebooks for rvq.
            ema_coef (float): exponential moving average coefficient for rvq training.
            steps_for_rvq_init (int): number of steps (batches) for rvq initialization.
        """
        super().__init__()

        self.encoder = Encoder(n_channels, latent_channels)
        self.rvq = RVQ(n_codebooks, codebook_size, latent_channels, ema_coef)
        self.decoder = Decoder(n_channels, latent_channels)

    def forward(self, audio, **batch):
        encoded_audio = self.encoder(audio)
        rvq_outputs = self.rvq(encoded_audio)
        reconstructed_audio = self.decoder(rvq_outputs["quantized"])
        return {
            "reconstructed_audio": reconstructed_audio,
            "encoded_audio": encoded_audio,
        } | rvq_outputs

    def rvq_done_initial_clustering(self):
        return self.rvq.done_initial_clustering.item()

    def rvq_init(self, X, num_iters):
        self.rvq.init_embeddings(X, num_iters)

    def forward_encoder_only(self, audio, **batch):
        return self.encoder(audio)

    def encode_to_indexes(self, audio):
        return self.rvq(self.encoder(audio))["rvq_indexes"]

    def decode_from_indexes(self, indexes):
        return self.decoder(self.rvq.decode_from_indexes(indexes))
