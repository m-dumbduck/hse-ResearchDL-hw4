import io

import requests
import soundfile
import torch
import torch.nn.functional as F
import torchaudio

# used 200 because encoder shrinks sequence by 5*5*4*2=200 times
MODEL_SHRINKAGE = 200


def load_mono_audio_from_url(url):
    response = requests.get(url)
    audio, sample_rate = soundfile.read(
        io.BytesIO(response.content), dtype="float32", always_2d=True
    )
    if audio.shape[1] != 1:
        audio = audio[:, :1]
    return audio, sample_rate


def resample_audio(audio, sample_rate, target_sample_rate):
    return torchaudio.functional.resample(
        torch.as_tensor(audio).T, orig_freq=sample_rate, new_freq=target_sample_rate
    ).T


def prepare_audio(audio, device):
    audio = torch.as_tensor(audio, dtype=torch.float32).permute([1, 0])
    raw_length = audio.shape[1]
    delta = (MODEL_SHRINKAGE - raw_length % MODEL_SHRINKAGE) % MODEL_SHRINKAGE
    audio = F.pad(audio, (0, delta), value=0.0)
    return audio.unsqueeze(0).to(device), raw_length


def post_process_audio(audio, length):
    return audio.squeeze(0).detach().cpu().numpy().T[:length]
