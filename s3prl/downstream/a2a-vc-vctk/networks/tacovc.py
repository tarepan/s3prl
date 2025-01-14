# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the simple (LSTMP), simple-AR and taco2-AR models for any-to-any voice conversion ]
#   Reference    [ `WaveNet Vocoder with Limited Training Data for Voice Conversion`, Interspeech 2018 ]
#   Reference    [ `Non-Parallel Voice Conversion with Autoregressive Conversion Model and Duration Adjustment`, Joint WS for BC & VCC 2020 ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""

from warnings import warn
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING

from .encoder import Taco2Encoder, ConfEncoder
from .conditioning import GlobalCondNet, ConfGlobalCondNet
from .decoder import Taco2Prenet, ConfDecoderPreNet, ExLSTMCell


@dataclass
class ConfDecoderMainNet:
    """
    Configuration of TacoVC Decoder MainNet.

    Args:
        dim_i_cond - Dimension size of conditioning input
        dim_i_ar - Dimension size of processed AR input
        dim_h - Dimension size of RNN hidden units
        num_layers - The number of RNN layers
        dropout_rate - RNN dropout rate
        layer_norm - Whether to use layer normalization in RNN
        projection - Whether LSTM or LSTMP
        dim_o - Dimension size of output
    """
    dim_i_cond: int = MISSING
    dim_i_ar: int = MISSING
    dim_h: int = MISSING
    num_layers: int = MISSING
    dropout_rate: float = MISSING
    layer_norm: bool = MISSING
    projection: bool = MISSING
    dim_o: int = MISSING

@dataclass
class ConfTacoVCNet:
    """
    Configuration of TacoVCNet.

    Args:
        dim_latent - Dimension size of latent between Encoder and Decoder
        dim_processed_ar - Dimension size of processed Decoder AR feature
        dim_o - Dimension size of output acoustic feature
    """
    dim_latent: int = MISSING
    dim_processed_ar: int = MISSING
    dim_o: int  = MISSING
    encoder: ConfEncoder = ConfEncoder(
        dim_o="${..dim_latent}",)
    global_cond: ConfGlobalCondNet = ConfGlobalCondNet(
        dim_io="${..dim_latent}",)
    dec_prenet: ConfDecoderPreNet = ConfDecoderPreNet(
        dim_i="${..dim_o}",
        dim_h_o="${..dim_processed_ar}",)
    dec_mainnet: ConfDecoderMainNet = ConfDecoderMainNet(
        dim_i_cond="${..dim_latent}",
        dim_i_ar="${..dim_processed_ar}",
        dim_o="${..dim_o}")

class TacoVCNet(nn.Module):
    """
    S3PRL-VC TacoVC network.

    `Taco2-AR`: segFC-3Conv-1LSTM-cat_(z_t, AR-segFC)-NuniLSTM-segFC-segLinear
    segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM-cat_(z_t, AR-norm-(segFC-ReLU-DO)xN)-(1uniLSTM[-LN][-DO]-segFC-Tanh)xL-segFC
    """

    def __init__(self,
        resample_ratio: float,
        conf: ConfTacoVCNet,
        mean: Optional[np.ndarray],
        scale: Optional[np.ndarray],
    ):
        """
        Args:
            resample_ratio - conditioning series resampling ratio
            conf - Configuration
            mean - FrequencyBand-wise mean
            scale - FrequencyBand-wise standard deviation
        """
        super().__init__()
        self.conf = conf
        self.resample_ratio = resample_ratio

        # Speaker-independent Encoder: segFC-Conv-LSTM // segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM
        self.encoder = Taco2Encoder(conf.encoder)

        # Global speaker conditioning network
        self.global_cond = GlobalCondNet(conf.global_cond)

        # Decoder
        ## PreNet: (segFC-ReLU-DO)xN
        if ((mean is None) and (scale is not None)) or ((mean is not None) and (scale is None)):
            raise Exception("Should be 'both mean/scale exist' OR 'both mean/scale not exist'")
        elif (mean is not None) and (scale is not None):
            self.register_spec_stat(mean, scale)
        self.prenet = Taco2Prenet(conf.dec_prenet)
        ## MainNet: LSTMP + linear projection
        conf = conf.dec_mainnet
        ### LSTMP: (1uniLSTM[-LN][-DO][-segFC-Tanh])xN
        self.lstmps = nn.ModuleList()
        for i_layer in range(conf.num_layers):
            # cat(local_cond, process(ar)) OR lower layer hidden_state/output
            dim_i_lstm = conf.dim_i_cond + conf.dim_i_ar if i_layer == 0 else conf.dim_h
            rnn_layer = ExLSTMCell(
                dim_i=dim_i_lstm,
                dim_h_o=conf.dim_h,
                dropout=conf.dropout_rate,
                layer_norm=conf.layer_norm,
                projection=conf.projection,
            )
            self.lstmps.append(rnn_layer)
        ### Projection: segFC
        self.proj = torch.nn.Linear(conf.dim_h, conf.dim_o)
        self.dim_o = conf.dim_o
        ## PostNet: None
        pass

    def register_spec_stat(self, mean: np.ndarray, scale: np.ndarray):
        """
        Register spectram statistics as model state.

        Args:
            mean -  frequencyBand-wise mean 
            scale - frequencyBand-wise standard deviation
        """
        # buffer is part of state_dict (saved by PyTorch functions)
        self.register_buffer("target_mean", torch.from_numpy(mean).float())
        self.register_buffer("target_scale", torch.from_numpy(scale).float())
        

    def forward(self, features, lens, spk_emb, targets = None):
        """Convert unit sequence into acoustic feature sequence.

        Args:
            features (Batch, T_max, Feat_i) - input unit sequences padded
            lens     (Batch,)               - Time length of each unit sequence
            spk_emb  (Batch, Spk_emb)       - speaker embedding vectors as global conditioning
            targets  (Batch, T_max, Feat_o) - padded target acoustic feature sequences
        Returns:
            ((Batch, Tmax, Freq), lens)
        """
        B = features.shape[0]

        # Resampling: resample the input features according to resample_ratio
        # (B, T_max, Feat_i) => (B, Feat_i, T_max) => (B, Feat_i, T_max') => (B, T_max', Feat_i)
        features = features.permute(0, 2, 1)
        # Nearest interpolation
        resampled_features = F.interpolate(features, scale_factor = self.resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self.resample_ratio

        # (resampled_features:(B, T_max', Feat_i)) -> (B, T_max', Feat_h)
        # `lens` is used for RNN padding. `si` stands for speaker-independent
        si_latent_series, lens = self.encoder(resampled_features, lens)

        # (B, T_max', Feat_h) -> (B, T_max', Feat_h)
        conditioning_series = self.global_cond(si_latent_series, spk_emb)

        # Decoder: spec_t' = f(spec_t, cond_t), cond_t == f(unit_t, spk_g)
        # AR decofing w/ or w/o teacher-forcing
        # Transpose for easy access: (B, T_max, Feat_o) => (T_max, B, Feat_o)
        if targets is not None:
            targets = targets.transpose(0, 1)
        predicted_list = []

        # Initialize LSTM hidden state and cell state of all LSTMP layers, and x_t-1
        _tensor = conditioning_series
        c_list = [_tensor.new_zeros(B, self.conf.dec_mainnet.dim_h)]
        z_list = [_tensor.new_zeros(B, self.conf.dec_mainnet.dim_h)]
        for _ in range(1, len(self.lstmps)):
            c_list += [_tensor.new_zeros(B, self.conf.dec_mainnet.dim_h)]
            z_list += [_tensor.new_zeros(B, self.conf.dec_mainnet.dim_h)]
        prev_out = _tensor.new_zeros(B, self.conf.dec_mainnet.dim_o)

        # step-by-step loop for autoregressive decoding
        ## local_cond::(B, hidden_dim)
        for t, local_cond in enumerate(conditioning_series.transpose(0, 1)):
            # Single time step
            ## RNN input (local conditioning and processed AR)
            ar = self.prenet(prev_out)
            cond_plus_ar = torch.cat([local_cond, ar], dim=1)
            ## Run single time step of all LSTMP layers
            for i, lstmp in enumerate(self.lstmps):
                # Run a layer (1uniLSTM[-LN][-DO]-segFC-Tanh), then update states
                # Input: RNN input OR lower layer's output
                lstmp_input = cond_plus_ar if i == 0 else z_list[i-1]
                z_list[i], c_list[i] = lstmp(lstmp_input, z_list[i], c_list[i])
            # Projection & Stack: Stack output_t `proj(o_lstmps)` in full-time list
            predicted_list += [self.proj(z_list[-1]).view(B, self.dim_o, -1)]
            # teacher-forcing if `target` else pure-autoregressive
            prev_out = targets[t] if targets is not None else predicted_list[-1].squeeze(-1)
            # AR spectrum is normalized (todo: could be moved up, but it change t=0 behavior)
            prev_out = (prev_out - self.target_mean) / self.target_scale
            # /Single time step
        # (Batch, Freq, 1?)[] -> (Batch, Freq, Tmax) -> (Batch, Tmax, Freq)
        predicted = torch.cat(predicted_list, dim=2).transpose(1, 2)

        return predicted, lens
