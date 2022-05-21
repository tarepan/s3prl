# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the any-to-any voice conversion downstream wrapper ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List

from scipy.io.wavfile import write
from tqdm import tqdm
import yaml

import torch
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from resemblyzer import preprocess_wav, VoiceEncoder
from omegaconf import OmegaConf, MISSING

from .networks.tacovc import TacoVCNet, ConfTacoVCNet
from .data.datamodule import WavMelEmbVcData, ConfWavMelEmbVcData
from .dataset import Stat

from .utils import make_non_pad_mask
from .utils import write_hdf5

FS = 16000

class Loss(nn.Module):
    """
    L1 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """
    def __init__(self, stats):
        """
        Args:
            stats (`Stat`): Mean and Scale statistics for normalization
        """
        super(Loss, self).__init__()
        self.objective = torch.nn.L1Loss(reduction="mean")
        self.register_buffer("target_mean", torch.from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", torch.from_numpy(stats.scale_).float())

    def normalize(self, x):
        return (x - self.target_mean) / self.target_scale

    def forward(self, x, y, x_lens, y_lens, device):
        """
        Args:
            x::Tensor[Batch, Tmax, Freq] - predicted_features
            y - acoustic_features_padded
            x_lens - predicted_feature_lengths
            y_lens - acoustic_feature_lengths
            device
        """
        # match the input feature length to acoustic feature length to calculate the loss
        if x.shape[1] > y.shape[1]:
            x = x[:, :y.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        if x.shape[1] <= y.shape[1]:
            y = y[:, :x.shape[1]]
            masks = make_non_pad_mask(x_lens).unsqueeze(-1).to(device)        

        x_normalized = self.normalize(x)
        y_normalized = self.normalize(y.to(device))

        # slice based on mask by PyTorch function
        x_masked = x_normalized.masked_select(masks)
        y_masked = y_normalized.masked_select(masks)

        loss = self.objective(x_masked, y_masked)
        return loss


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_warmup_step: The number of LR shaduler warmup steps
        sched_total_step: The number of total training steps
    """
    learning_rate: float = MISSING
    sched_warmup_step: int = MISSING
    sched_total_step: int = MISSING


from dataclasses import dataclass

@dataclass
class ConfRunner:
    """
    Args:
        expdir - Directory in which dev/test results are saved
    """
    dim_mel: int = MISSING
    train_steps: int = MISSING
    net: ConfTacoVCNet = ConfTacoVCNet()
    optim: ConfOptim = ConfOptim()
    data: ConfWavMelEmbVcData = ConfWavMelEmbVcData()
    expdir: str = MISSING

class DownstreamExpert(nn.Module):
    """S3PRL interface of a2a-vc-vctk
    """

    def __init__(self, upstream_dim: int, upstream_rate, downstream_expert, _, **kwargs):
        """
        Args:
            upstream_dim - Feature dimension size of upstream output
            upstream_rate
            downstream_expert - the `downstream_expert` attribute of config
        """
        super().__init__()

        # conf generation (deleted in PL)
        self.conf = OmegaConf.merge(
            OmegaConf.structured(ConfRunner),
            OmegaConf.create(downstream_expert)
        )

        # Datasets (moved in PL)
        self._data = WavMelEmbVcData(self.conf.data)
        self._data.prepare_data()
        self._data.setup()

        # Time-directional up/down sampling ratio toward input series
        _fs = self.conf.data.dataset.sr_for_mel
        n_shift = self.conf.data.dataset.n_shift
        resample_ratio = _fs / n_shift * upstream_rate / FS
        print(f'[Downstream] - resample_ratio: {resample_ratio}')

        # Load dataset-wise statistics
        # todo: Is this (AR normalization besed on only trainings, over speaker) good?
        stats = self._data.dataset_train.acquire_spec_stat()

        # define model and loss
        self.model = TacoVCNet(
            resample_ratio=resample_ratio,
            stats=stats,
            conf=conf.net,
        )
        self.objective = Loss(stats)
        # Utterance embedding model for inference
        self.uttr_encoder = None

    # Interface
    def forward(self,
                split,
                input_features,
                acoustic_features,
                acoustic_features_padded,
                acoustic_feature_lengths,
                spk_embs,
                vc_ids,
                records,
                **kwargs):
        """S3PRL interface for calculation.

        Args:
            split: mode
            input_features: list of unpadded features generated by the upstream
            acoustic_features: List[Tensor(`lmspc`)], not used...?
            acoustic_features_padded: `acoustic_features` padded by PyTorch function
            acoustic_feature_lengths: Tensor(feature time length)
            spk_embs: Tensor(`ref_spk_emb`)
            vc_ids: List[(target_spk, source_spk, uttr_name)]
        """

        device = input_features[0].device
        spk_embs = spk_embs.to(device)
        acoustic_features_padded = acoustic_features_padded.to(device)

        # input_feature_lengths::(B, T)
        input_feature_lengths = torch.IntTensor([feature.shape[0] for feature in input_features])
        # (T, Feat)[] -> (B, Tmax, Feat)
        input_features = pad_sequence(input_features, batch_first=True).to(device=device)

        if split == "train":
            results = self.training_step((
                input_features, input_feature_lengths, \
                acoustic_features_padded, acoustic_feature_lengths, \
                spk_embs, device,
                ), 1
            )
            loss = results["loss"]
        # eval | test
        else:
            results = self.validation_step((
                input_features, input_feature_lengths, \
                acoustic_features_padded, acoustic_feature_lengths, \
                spk_embs, device, records
                ), 2
            )
            loss = results["val_loss"]
            records["vc_ids"] += vc_ids

        records['loss'].append(loss.item())
        return loss

    def training_step(self, batch, batch_idx: int):
        """Forward a batch.

        Args:
            batch
                input_features
                input_feature_lengths
                acoustic_features_padded
                acoustic_feature_lengths
                spk_embs - Speaker embeddings
                device - device
            batch_idx - Batch index in a training epoch
        Returns - loss
        """
        input_features, input_feature_lengths, \
            acoustic_features_padded, acoustic_feature_lengths, \
            spk_embs, device = batch

        # The forward
        predicted_features, predicted_feature_lengths = self.model(
            input_features, input_feature_lengths, \
            spk_embs,
            acoustic_features_padded,
        )

        # Masked/normalized L1 loss
        loss = self.objective(predicted_features,
                              acoustic_features_padded,
                              predicted_feature_lengths,
                              acoustic_feature_lengths,
                              device)

        # self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        input_features, input_feature_lengths, \
            acoustic_features_padded, acoustic_feature_lengths, \
            spk_embs, device, records = batch

        predicted_features, predicted_feature_lengths = self.model(
            input_features,
            input_feature_lengths,
            spk_embs,
        )
        # Masked/normalized L1 loss
        loss = self.objective(predicted_features,
                            acoustic_features_padded,
                            predicted_feature_lengths,
                            acoustic_feature_lengths,
                            device)
        # self.log("val_loss", loss)

        # save the unnormalized features for dev and test sets
        records["predicted_features"] += predicted_features.cpu().numpy().tolist()
        records["feature_lengths"] += predicted_feature_lengths.cpu().numpy().tolist()

        # todo: Synthesis
        pass

        # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.
        #     utils.tensorboard.writer.SummaryWriter.add_audio)
        # self.logger.experiment.add_audio(
        #     f"audio_{batch_idx}",
        #     wave, # snd_tensor: Tensor(1, L)
        #     global_step=self.global_step,
        #     sample_rate=self.conf.sampling_rate,
        # )

        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx: int = 0):
        """Generate a mel-spectrogram from a unit sequence and speaker embedding.
        Args:
            batch
                unit_series::Tensor[Batch==1, TimeUnit, Feat] - Input unit sequence
                target_emb::Tensor[Batch==1, Emb] - Target style embedding
        Returns:
            Tensor[Batch==1, TimeMel, Freq] - mel-spectrogram
        """
        unit_series, target_emb = batch
        return self.model(unit_series.to(self.device), target_emb.to(self.device))

    def configure_optimizers(self):
        """Set up a optimizer
        """

        optim = AdamW(self.model.parameters(), lr=self.conf.optim.learning_rate)

        # Scheduler's multiplicative factor function
        total_steps = self.conf.optim.sched_total_step
        warmup_steps = self.conf.optim.sched_warmup_step
        def lr_lambda(now: int) -> float:
            """0@0 ---> (linear) ---> 1@`warmup_steps` ---> (linear) ---> 0@`total_steps`"""
            is_warmup = now < warmup_steps
            return (now / warmup_steps) if is_warmup else (total_steps - now) / (total_steps - warmup_steps)

        sched = {
            "scheduler": LambdaLR(optim, lr_lambda),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }

    def mel_taco_to_rnnms(self, log_amp_bel: torch.Tensor) -> torch.Tensor:
        """
        Convert TacoVC-compatible mel-spectrogram to RNNMS-compatible one.

        Args:
            log_amp_bel::[Batch==1, TimeMel, Freq] - log(ref=0dB, min=-200dB)-amplitude [B]
        Returns:
            rnnms_mel::[Batch==1, TimeMel, Freq] - scaled(1/80)-log(ref=20dB, minrel=-80dB)-power
        """
        log_amp_dB = 10. * log_amp_bel # log(ref=0dB, min=-200dB)-amplitude [dB]
        log_pow = 2. * log_amp_dB      # log(S^2/1) = 2*log(S/1) ==> 10*log(S^2/1) [dB] = 10*2*log(S/1) = 2*(10*log(S/1))
        log_pow_ref20 = log_pow - 20.
        log_pow_ref20_minrel80 = torch.maximum(torch.tensor([-80.]), log_pow_ref20)
        return log_pow_ref20_minrel80 / 80.

    def wavs2emb(self, waves: List[np.ndarray]) -> torch.Tensor:
        """Convert waveforms into an averaged embedding.

        Args:
            waves::List[(Time,)] - waveforms, each of which can have different length
        Returns:
            ave_emb::Tensor[Batch=1, Emb] - an averaged embedding
        """

        # Initialization at first call
        if self.uttr_encoder is None:
            self.uttr_encoder = VoiceEncoder().to(self.device)

        # Calculate an average of utterance embeddings
        processed_waves = [preprocess_wav(wave) for wave in waves]
        ave_emb = self.uttr_encoder.embed_speaker(processed_waves)
        
        return torch.unsqueeze(torch.from_numpy(ave_emb), 0)

    # interface
    def log_records(self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        """S3PRL interface for logging.

        Report loss, save features and generated-waveform

        Args:
            split: `dev` or `test`?
            records: Logging target record
                loss
                feature_lengths
                predicted_features
                vc_ids
            logger: (maybe) TensorBoard logger
            global_step: Number of global step, used for file name and TB logging
            batch_ids: Not Used
            total_batch_num: Not Used
            kwargs: Not Used
        Returns:
            empty array
        """

        # Loss logging in console and TB
        loss = torch.FloatTensor(records['loss']).mean().item()
        print(f'{split} loss: {loss:.6f}')
        logger.add_scalar(f'example/{split}', loss, global_step=global_step)

        # Generate waveform w/ Griffin-Lim and save it
        if split in ["dev", "test"]:
            # Path preparation
            root = Path(self.conf.expdir) / str(global_step) / split
            hdf5_save_dir = root / "hdf5"
            wav_save_dir = root  / "wav"
            hdf5_save_dir.mkdir(exist_ok=True, parents=True)
            wav_save_dir.mkdir(exist_ok=True, parents=True)

            for i, (tgt_spk, src_spk, uttr_name) in enumerate(tqdm(
                records["vc_ids"],
                dynamic_ncols=True, desc="Inference/Generate_waveform"
            )):
                # Save each item in `predicted_features` w/o contents modification
                # No.i in a batch
                length = int(records["feature_lengths"][i])
                # Remove padding:: (T_mel, Freq)
                fbank = np.array(records["predicted_features"][i])[:length]

                # Path preparation
                file_stem = f"{tgt_spk}_from_{src_spk}_{uttr_name}"
                hdf5_save_path = hdf5_save_dir / f"{file_stem}.h5"
                wav_save_path = wav_save_dir / f"{file_stem}.wav"

                # save generated features into hdf5 files
                write_hdf5(hdf5_save_path, "feats", fbank)

                # Waveform generation from feature for reporting
                # wav = vocoder(fbank)
                ## save
                # write(
                #     wav_save_path,
                #     <sampling_rate>,
                #     (y * np.iinfo(np.int16).max).astype(np.int16),
                # )
        return []

    @property
    def device(self):
        return next(self.parameters()).device

    # Interface
    def get_dataloader(self, split: str) -> DataLoader:
        """(S3PRL API) Generate data loader."""
        if split == "train":
            return self._data.train_dataloader()
        elif split == "dev":
            return self._data.val_dataloader()
        elif split == "test":
            return self._data.test_dataloader()
