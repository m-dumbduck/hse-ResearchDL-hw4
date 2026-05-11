from torch import nn

from src.model.base_model import BaseModel


class ResidualUnit2D(BaseModel):
    """
    Residual Unit 2D for STFT Discriminator
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride_t: int, stride_f: int
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(stride_t + 2, stride_f + 2),
                stride=(stride_t, stride_f),
                padding=1,
            ),
        )
        self.projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=(stride_t, stride_f),
        )

    def forward(self, X):
        res = self.net(X)
        return res + self.projector(X)[:, :, : res.shape[2], : res.shape[3]]
