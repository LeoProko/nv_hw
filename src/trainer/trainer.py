import random

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import torchaudio

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.loss import hifi as hifi_loss
from src.utils import make_mel
from src.loss import hifi as hifi_loss


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        metrics,
        gen_optimizer,
        desc_optimizer,
        config,
        device,
        dataloaders,
        gen_lr_scheduler=None,
        desc_lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, metrics, gen_optimizer, desc_optimizer, config, device)
        self.skip_oom = skip_oom
        self.device = device
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.wav2mel = make_mel.MelSpectrogram()
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.gen_lr_scheduler = gen_lr_scheduler
        self.desc_lr_scheduler = desc_lr_scheduler
        self.log_step = 50

        self.valid_audios_spec = [
            self.wav2mel(self.load_audio(f"data/valid/audio_{i}.wav"))
            for i in range(1, 3 + 1)
        ]

        self.train_metrics = MetricTracker(
            "desc_loss",
            "gen_loss",
            "grad norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            "desc_loss",
            "gen_loss",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in [
            "audio",
            "spectrogram",
        ]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, db in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            # for db in batch:
            try:
                db = self.process_batch(
                    db,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {}".format(
                        epoch,
                        self._progress(batch_idx),
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.gen_lr_scheduler.get_last_lr()[0]
                )
                self.model.eval()
                self._log_predictions()
                self.model.train()
                # self._log_spectrogram(db["mel_output"].detach())
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        if is_train:
            self.desc_optimizer.zero_grad()

        generated_wav = self.model.generator(batch["spectrogram"])
        generated_mel = self.wav2mel(generated_wav.detach().cpu())

        target_msd_outputs, gen_msd_outputs, _, _ = self.model.msd(
            batch["audio"], generated_wav
        )
        target_mpd_outputs, gen_mpd_outputs, _, _ = self.model.mpd(
            batch["audio"], generated_wav
        )

        desc_msd_loss = hifi_loss.discriminator_loss(
            target_msd_outputs, gen_msd_outputs
        )
        desc_mpd_loss = hifi_loss.discriminator_loss(
            target_mpd_outputs, gen_mpd_outputs
        )
        metrics.update("desc_loss", (desc_msd_loss + desc_mpd_loss).item())

        if is_train:
            (desc_msd_loss + desc_mpd_loss).backward()
            self._clip_grad_norm()
            self.desc_optimizer.step()

            self.gen_optimizer.zero_grad()

        generated_wav = self.model.generator(batch["spectrogram"])
        generated_mel = self.wav2mel(generated_wav.detach().cpu())

        (
            target_msd_outputs,
            gen_msd_outputs,
            target_msd_fms,
            gen_msd_fms,
        ) = self.model.msd(batch["audio"], generated_wav)

        (
            target_mpd_outputs,
            gen_mpd_outputs,
            target_mpd_fms,
            gen_mpd_fms,
        ) = self.model.mpd(batch["audio"], generated_wav)

        spec_loss = hifi_loss.spec_loss(generated_mel, batch["spectrogram"])
        desc_msd_loss = hifi_loss.feature_loss(target_msd_fms, gen_msd_fms)
        desc_mpd_loss = hifi_loss.feature_loss(target_mpd_fms, gen_mpd_fms)
        gen_msd_loss = hifi_loss.generator_loss(gen_msd_outputs)
        gen_mpd_loss = hifi_loss.generator_loss(gen_mpd_outputs)

        if is_train:
            (
                spec_loss + desc_msd_loss + desc_mpd_loss + gen_msd_loss + gen_mpd_loss
            ).backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()

            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.step()
            if self.desc_lr_scheduler is not None:
                self.desc_lr_scheduler.step()

        metrics.update(
            "gen_loss",
            (
                spec_loss + desc_msd_loss + desc_mpd_loss + gen_msd_loss + gen_mpd_loss
            ).item(),
        )

        if is_train:
            for met in self.metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions()
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.config["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def _log_predictions(self):
        for i in range(len(self.valid_audios_spec)):
            gen_audio = self.model.generator(self.valid_audios_spec[i].to(self.device))
            self.writer.add_audio(
                f"valid-audio-{i}",
                gen_audio.squeeze(),
                sample_rate=self.config["sr"],
            )

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
