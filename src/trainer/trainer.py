import torch
import torchaudio
from matplotlib import pyplot as plt
from sympy.solvers.diophantine.diophantine import reconstruct
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)

            generator_outputs = self.generator(**batch)
            batch.update(generator_outputs)

            # Discriminator step
            discriminator_outputs = self.discriminator.forward_for_both(
                audio=batch["audio"],
                reconstructed_audio=batch["reconstructed_audio"].detach(),
            )
            batch.update(discriminator_outputs)
            discriminator_loss = self.discriminator_criterion(**batch)
            batch.update(discriminator_loss)

            batch["discriminator_loss"].backward()
            self._clip_grad_norm_discriminator()
            self.discriminator_optimizer.step()
            if self.discriminator_lr_scheduler is not None:
                self.discriminator_lr_scheduler.step()

            # Generator step
            for p in self.discriminator.parameters():
                p.requires_grad_(False)

            with torch.no_grad():
                batch["discriminator_for_audio"] = self.discriminator(
                    audio=batch["audio"]
                )
            batch["discriminator_for_reconstructed_audio"] = self.discriminator(
                audio=batch["reconstructed_audio"]
            )

            generator_loss = self.generator_criterion(**batch)
            batch.update(generator_loss)

            batch["generator_loss"].backward()
            self._clip_grad_norm_generator()
            self.generator_optimizer.step()
            if self.generator_lr_scheduler is not None:
                self.generator_lr_scheduler.step()

            for p in self.discriminator.parameters():
                p.requires_grad_(True)

            # update metrics for each loss (in case of multiple losses)
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name].item())
        else:
            metric_funcs = self.metrics["inference"]
            generator_outputs = self.generator(**batch)
            batch.update(generator_outputs)

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        audio = batch["audio"][0]
        reconstructed_audio = batch["reconstructed_audio"][0]
        sample_rate = int(batch["sample_rate"][0])

        self.writer.add_audio(
            f"{mode}/audio/original",
            audio=audio,
            sample_rate=sample_rate,
        )

        self.writer.add_audio(
            f"{mode}/audio/reconstructed",
            audio=reconstructed_audio,
            sample_rate=sample_rate,
        )

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=self.config.trainer.log_mel.win_length,
            n_fft=self.config.trainer.log_mel.win_length,
            hop_length=self.config.trainer.log_mel.hop_length,
            n_mels=self.config.trainer.log_mel.n_mels,
        ).to(self.device)

        self.writer.add_image(
            f"{mode}/mel/original", self._mel_to_image(mel_transform(audio))
        )

        self.writer.add_image(
            f"{mode}/mel/reconstructed",
            self._mel_to_image(mel_transform(reconstructed_audio)),
        )

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            pass
        else:
            pass

    def _mel_to_image(self, mel):
        mel = mel.detach().cpu().squeeze(0)
        mel = torch.log(mel + 1e-12).numpy()
        return plt.get_cmap("magma")(mel)[:, :, :3]

    def _on_train_start(self):
        self.init_rvq_codebooks()

    @torch.no_grad()
    def init_rvq_codebooks(self):
        if self.generator.rvq_done_initial_clustering():
            return

        self.logger.info("Initializing RVQ codebooks...")
        self.generator.eval()
        encoded_batches = []
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader_raw, total=self.config.trainer.rvq_init_steps)
        ):
            if batch_idx >= self.config.trainer.rvq_init_steps:
                break
            batch = self.move_batch_to_device(batch)
            batch = self.transform_batch(batch)
            encoded_audio = self.generator.forward_encoder_only(**batch)
            encoded_batches.append(encoded_audio.detach())

        encoded_audio = torch.cat(encoded_batches, dim=0)
        self.generator.rvq_init(
            encoded_audio, num_iters=self.config.trainer.get("rvq_kmeans_num_iters", 5)
        )

        self.generator.train()
