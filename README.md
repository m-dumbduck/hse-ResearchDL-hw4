# SoundStream Codec

This repository contains implementation of a [SoundStream](https://arxiv.org/abs/2107.03312)-style neural audio codec
for the HSE DL Research homework 4.

The model operates only on 16kHz audio. FiLM and bitrate dropout are not
used in this implementation.

The model consists of:

- Generator
  - convolutional encoder
  - residual vector quantization (RVQ)
  - convolutional decoder
- Discriminator
  - Waveform discriminators
  - STFT discriminator

For training we used the following objectives:

- Generator:
  - adversarial
  - feature matching
  - multiscale reconstruction
  - commitment
- Discriminator:
  - adversarial

All of them except for the `commitment` are described in the paper.
As `commitment` objective we used MSE between encoder output and its
quantized detached version.

The model shrinks time dimension $200$ times.
RVQ uses $8$ codebooks, each sized $1024=2^{10}$. Hence it needs $80$ bits
per timestep. If we compress $1$s audio in $16$kHz sample rate we will get
$16000 / 200 = 80$ timesteps per second, so the implemented codec compresses
audio to $6.4$kbps.


### Dataset

For training and evaluation we used [LibriSpeech dataset](https://www.openslr.org/12)
`train-clean-100` and `test-clean` partitions.

If you want to reproduce the experiments in this project, download the
required LibriSpeech partitions into `data/LibriSpeech`.

### Training and Evaluation

The model was trained on LibriSpeech dataset partition `train-clean-100`.
We provide a [Comet report](https://www.comet.com/m-dumbduck/hse-dl-hw-4/reports/C64vMA19MISqrsl5VxYoLujxa)
with logs for train and evaluation + detailed analysis.

Trained generator is published on HuggingFace: [ldiujes/SoundStream](https://huggingface.co/ldiujes/SoundStream)

Final Evaluation on `LibriSpeech test-clean`:

- `STOI = 0.81`
- `NISQA_MOS = 2.51`



## Installation

Basic installation steps are the following:

```bash
git clone https://github.com/m-dumbduck/hse-ResearchDL-hw4.git
cd hse-ResearchDL-hw4

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## Demo notebook

We provide a fully standalone `demo.ipynb` notebook containing
trained model performance demonstration. It loads audio from
a given custom URL (default URL is provided), loads
[trained model](https://huggingface.co/ldiujes/SoundStream),
runs model and demonstrates audio`s reconstructed version.

Before launching the notebook you may want to create virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

There are two scenarios in which notebook can be used.

1. **Notebook is installed on its own.** In this case by running its cells the notebook clones
the repository to a folder near itself. You may want to
use this scenario if you want to inference our model in Colab/Kaggle.
2. **Notebook is installed with repository.** In this case
notebook will also work properly. Running its cells will not trigger any repository cloning.


## Training

Run training with the default Hydra config:

```bash
python train.py
```

This uses:

- `src/configs/train.yaml`
- `src/model/sound_stream/` generator and discriminator architectures
- LibriSpeech `train-clean-100` for training
- LibriSpeech `test-clean` for evaluation
- CometML logging: `losses`, `NISQA_MOS`, `STOI`, `Codebook perplexity` etc

In the hydra training config you can adjust model hyperparameters, losses
hyperparameters, logging frequency, RVQ init hyperparameters etc.
You can use `resume_from` option to resume training from saved checkpoint in case
your training suddenly stopped.

## Inference and Evaluation

To run reconstruction and metric evaluation on test-clean, you can use a full training checkpoint or
HuggingFace repo id (for example `ldiujes/SoundStream`) containing a trained model. By default `inferencer.yaml` is set to load our pretrained
model from [ldiujes/SoundStream](https://huggingface.co/ldiujes/SoundStream).

```bash
python inference.py
```

Pretrained model is specified by
`from_pretrained_type` and `from_pretrained` config fields. `from_pretrained_type` must be 'hf' or 'file' and it specifies the regime which is used to load pretrained
model: `hf` - for HuggingFace HF PytorchMixin models. You have to pass a link to the
HuggingFace repo, `file` - for local checkpoint.


## Export to Hugging Face

You may want to export generator from a full training checkpoint to Hugging Face.
Lets assume you have trained a model for at least $10$ epochs. Then you can do:

```bash
python export_generator.py \
  --checkpoint saved/train/checkpoint-epoch10.pth \
  --repo-id your_hf_nickname/your_hf_repo_name
```

This will push trained generator from a given config to Hugging Face
using PytorchModelHubMixin.

After export, the model can be loaded with:

```python
from src.model.sound_stream import Generator

generator = Generator.from_pretrained("ldiujes/SoundStream")
```

Note: from_pretrained option in configs does not use `src.model.sound_stream.Generator`.
It downloads the full repo and uses safetensors weights from it. The type of inference the snippet
shows is used in `demo.ipynb`.

## Credits

This repository is based on pytorch project template [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).


## Used links
- [Original SoundStream paper](https://arxiv.org/abs/2107.03312)
- [GitHub repository](https://github.com/m-dumbduck/hse-ResearchDL-hw4)
- [Hugging Face model](https://huggingface.co/ldiujes/SoundStream)
- [Comet report](https://www.comet.com/m-dumbduck/hse-dl-hw-4/reports/C64vMA19MISqrsl5VxYoLujxa)


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
