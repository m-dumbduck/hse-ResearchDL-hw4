import torch
from huggingface_hub import hf_hub_download

from src.datasets.collate import collate_fn
from src.model.sound_stream import Generator
from src.transforms import Compose, Permute, ResampleAudio, ToTensor

SAMPLE_RATE = 16000
REPO_ID = "ldiujes/SoundStream"
FILENAME = "soundstream_generator.pth"


def load_soundstream_generator(device="cpu"):
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    generator = Generator(
        n_channels=32,
        latent_channels=512,
        n_codebooks=8,
        codebook_size=1024,
        ema_coef=0.99,
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state_dict)
    return generator


def prepare_audio(audio, sample_rate, device):
    item = {
        "audio": audio,
        "sample_rate": sample_rate,
    }
    transforms = Compose(
        [
            ToTensor(data_object_key="audio"),
            Permute(data_object_key="audio", order=[1, 0]),
            ResampleAudio(
                data_object_key="audio",
                sample_rate_key="sample_rate",
                target_sample_rate=SAMPLE_RATE,
            ),
        ]
    )
    item = transforms(item)
    item["audio"] = item["audio"].to(device)
    return collate_fn([item])


@torch.no_grad()
def run_inference(generator, audio, sample_rate, device="cpu"):
    generator = generator.to(device).eval()

    batch = prepare_audio(
        audio=audio,
        sample_rate=sample_rate,
        device=device,
    )

    outputs = generator(**batch)
    batch.update(outputs)

    raw_length = batch["raw_length"][0]
    original_audio = batch["audio"][0, :, :raw_length].cpu().numpy().T
    reconstructed_audio = (
        batch["reconstructed_audio"][0, :, :raw_length].cpu().numpy().T
    )
    rvq_indexes = batch["rvq_indexes"].cpu().numpy()

    return {
        "original_audio": original_audio,
        "reconstructed_audio": reconstructed_audio,
        "rvq_indexes": rvq_indexes,
    }
