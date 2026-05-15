import warnings
from pathlib import Path

import hydra
import torch
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the generator, discriminator, generator_optimizer,
    discriminator_optimizer, generator_scheduler, discriminator_scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    if config.trainer.get("from_pretrained_type") == "hf":
        config.trainer.from_pretrained = str(
            Path(snapshot_download(repo_id=config.trainer.from_pretrained))
            / "model.safetensors"
        )

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build models architecture, then print to console
    generator = instantiate(config.generator.model).to(device)
    discriminator = instantiate(config.discriminator.model).to(device)
    logger.info(generator)
    logger.info(discriminator)

    # get function handles of loss and metrics
    generator_loss_function = instantiate(config.generator.loss_function).to(device)
    discriminator_loss_function = instantiate(config.discriminator.loss_function).to(
        device
    )
    metrics = instantiate(config.metrics)

    # build optimizers, learning rate schedulers
    generator_trainable_params = filter(
        lambda p: p.requires_grad, generator.parameters()
    )
    generator_optimizer = instantiate(
        config.generator.optimizer, params=generator_trainable_params
    )
    generator_lr_scheduler = instantiate(
        config.generator.lr_scheduler, optimizer=generator_optimizer
    )
    discriminator_trainable_params = filter(
        lambda p: p.requires_grad, discriminator.parameters()
    )
    discriminator_optimizer = instantiate(
        config.discriminator.optimizer, params=discriminator_trainable_params
    )
    discriminator_lr_scheduler = instantiate(
        config.discriminator.lr_scheduler, optimizer=discriminator_optimizer
    )

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        generator_criterion=generator_loss_function,
        discriminator_criterion=discriminator_loss_function,
        metrics=metrics,
        generator_optimizer=generator_optimizer,
        generator_lr_scheduler=generator_lr_scheduler,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        config=config,
        pretrained_type=config.trainer.get("from_pretrained_type"),
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
