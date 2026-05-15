import io

import requests
import soundfile


def load_audio_from_url(url):
    response = requests.get(url)
    audio, sample_rate = soundfile.read(
        io.BytesIO(response.content), dtype="float32", always_2d=True
    )
    if audio.shape[1] != 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio, sample_rate
