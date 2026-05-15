import io

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image


def mel_comparison_plot(mel_original, mel_reconstructed):
    mel_original = torch.log(mel_original.squeeze(0) + 1e-12).numpy()
    mel_reconstructed = torch.log(mel_reconstructed.squeeze(0) + 1e-12).numpy()
    vmin = min(mel_original.min(), mel_reconstructed.min())
    vmax = max(mel_original.max(), mel_reconstructed.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    axes[0].imshow(
        mel_original,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Mel bins")

    axes[1].imshow(
        mel_reconstructed,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Reconstructed")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mel bins")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))
