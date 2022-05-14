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

from scipy.io.wavfile import write
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model, ConfModel
from .dataset import VCTK_VCC2020Dataset, Stat
from .utils import make_non_pad_mask
from .utils import write_hdf5
from .utils import logmelspc_to_linearspc, griffin_lim

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


class DownstreamExpert(nn.Module):
    """S3PRL interface of a2a-vc-vctk

    Dataset: `VCTK_VCC2020Dataset` (train/dev/test)
    """

    def __init__(self, upstream_dim:int, upstream_rate, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: Feature dimension size of upstream output
            upstream_rate:
        """
        super(DownstreamExpert, self).__init__()
        
        # basic settings
        self.expdir = expdir
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.acoustic_feature_dim = self.datarc["fbank_config"]["n_mels"]
        _fs = self.datarc["fbank_config"]["fs"]
        # Time-directional up/down sampling ratio toward input series
        self.resample_ratio = _fs / self.datarc["fbank_config"]["n_shift"] * upstream_rate / FS
        print('[Downstream] - resample_ratio: ' + str(self.resample_ratio))

        # Load datasets
        self.train_dataset = VCTK_VCC2020Dataset('train', **self.datarc)
        self.dev_dataset = VCTK_VCC2020Dataset('dev', **self.datarc)
        self.test_dataset = VCTK_VCC2020Dataset('test', **self.datarc)

        # Load dataset-wise statistics
        # todo: Is this (AR normalization besed on only trainings, over speaker) good?
        self.stats = self.train_dataset.acquire_spec_stat()

        # Model configuration
        conf = ConfModel(
            dim_latent=self.modelrc.hidden_dim,
            encoder=ConfEncoder(
                dim_i=self.upstream_dim,
                causal=self.modelrc.enc_conv_causal,
                bidirectional=self.modelrc.enc_bidi,
                dim_o=self.modelrc.hidden_dim,),
            global_cond=ConfGlobalCondNet(
                integration_type=self.modelrc.spk_emb_integration_type,
                dim_io=self.modelrc.hidden_dim,
                dim_global_cond=self.modelrc.spk_emb_dim,),
            # dec.dim_ar=,
            # dec.dim_o=self.acoustic_feature_dim
            dec_prenet=ConfDecoderPreNet(
                dim_i=self.acoustic_feature_dim,
                dim_h_o=self.modelrc.prenet_dim,
                n_layers=self.modelrc.prenet_layers,
                dropout_rate=self.modelrc.prenet_dropout_rate,),
            dec_mainnet=ConfDecoderMainNet(
                dim_i_cond=self.modelrc.hidden_dim,
                dim_i_ar=self.modelrc.prenet_dim,
                dim_h=self.modelrc.hidden_dim,
                num_layers=self.modelrc.lstmp_layers,
                dropout_rate=self.modelrc.lstmp_dropout_rate,
                layer_norm=self.modelrc.lstmp_layernorm,
                projection=True,
                dim_o=self.acoustic_feature_dim,),
        )

        # define model and loss
        self.model = Model(
            resample_ratio=self.resample_ratio,
            stats=self.stats,
            conf=conf,
        )
        self.objective = Loss(self.stats)

    # Interface
    def get_dataloader(self, split):
        """S3PRL interface for data load"""
        if split == 'train':
            return self._get_train_dataloader(self.train_dataset)
        elif split == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)
        elif split == 'test':
            return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, dataset):
        """collate_fn should be implemented as dataset's method"""
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        """collate_fn should be implemented as dataset's method"""
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

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
            loss = self.training_step((
                input_features, input_feature_lengths, \
                acoustic_features_padded, acoustic_feature_lengths, \
                spk_embs, device,
                ), 1
            )
        else:
            loss = self.validation_step((
                input_features, input_feature_lengths, \
                acoustic_features_padded, acoustic_feature_lengths, \
                spk_embs, device, records
                ), 2
            )
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
        return loss

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

        return loss

    def predict_step(self, batch, batch_idx: int):
        # unit_series, spk_emb = batch
        # return self.model(unit_series, spk_emb)
        pass

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
            root = Path(self.expdir) / str(global_step) / split
            hdf5_save_dir = root / "hdf5"
            wav_save_dir = root  / "wav"
            hdf5_save_dir.mkdir(exist_ok=True, parents=True)
            wav_save_dir.mkdir(exist_ok=True, parents=True)

            for i, (tgt_spk, src_spk, uttr_name) in enumerate(tqdm(
                records["vc_ids"],
                dynamic_ncols=True, desc="Inference/Generate_waveform"
            )):
                # No.i in a batch
                length = int(records["feature_lengths"][i])
                fbank = np.array(records["predicted_features"][i])[:length]

                # Path preparation
                file_stem = f"{tgt_spk}_from_{src_spk}_{uttr_name}"
                hdf5_save_path = hdf5_save_dir / f"{file_stem}.h5"
                wav_save_path = wav_save_dir / f"{file_stem}.wav"

                # save generated features into hdf5 files
                write_hdf5(hdf5_save_path, "feats", fbank)

                # Waveform generation from feature for reporting
                # mel_spec => linear_spec => (Griffin-Lim) => waveform
                ## mel fbank => linear spectrogram
                spc = logmelspc_to_linearspc(
                    fbank,
                    fs=self.datarc["fbank_config"]["fs"],
                    n_mels=self.datarc["fbank_config"]["n_mels"],
                    n_fft=self.datarc["fbank_config"]["n_fft"],
                    fmin=self.datarc["fbank_config"]["fmin"],
                    fmax=self.datarc["fbank_config"]["fmax"],
                )
                ## linear spectrogram -> Griffin-Lim -> waveform
                y = griffin_lim(
                    spc,
                    n_fft=self.datarc["fbank_config"]["n_fft"],
                    n_shift=self.datarc["fbank_config"]["n_shift"],
                    win_length=self.datarc["fbank_config"]["win_length"],
                    window=self.datarc["fbank_config"]["window"],
                    n_iters=self.datarc["fbank_config"]["gl_iters"],
                )
                ## save
                write(
                    wav_save_path,
                    self.datarc["fbank_config"]["fs"],
                    (y * np.iinfo(np.int16).max).astype(np.int16),
                )
        return []
