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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Stat
from .networks.encoder import Taco2Encoder, ConfEncoder
from .networks.conditioning import GlobalCondNet, ConfGlobalCondNet
from .networks.decoder import Taco2Prenet, ConfDecoderPreNet, ExLSTMCell



################################################################################

@dataclass
class ConfDecoderMainNet:
    dim_i_cond: int # Dimension size of conditioning input
    dim_i_ar: int   # Dimension size of processed AR input
    dim_h: int    # Dimension size of RNN hidden units
    num_layers: int
    dropout_rate: float
    layer_norm: bool
    projection: bool
    dim_o: int    # Dimension size of output

@dataclass
class ConfModel:
    """Configuration of Model"""
    hidden_dim: int       # Dimension size of latent, equal to o_encoder and i_decoder (preserved for future local sync)
    encoder: ConfEncoder
    global_cond: ConfGlobalCondNet
    dec_prenet: ConfDecoderPreNet
    dec_mainnet: ConfDecoderMainNet

class Model(nn.Module):
    """
    S3PRL-VC model.

    `Taco2-AR`: segFC-3Conv-1LSTM-cat_(z_t, AR-segFC)-NuniLSTM-segFC-segLinear
    segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM-cat_(z_t, AR-norm-(segFC-ReLU-DO)xN)-(1uniLSTM[-LN][-DO]-segFC-Tanh)xL-segFC
    """

    def __init__(self, resample_ratio, stats: Stat, conf: ConfModel):
        """
        Args:
            stats (`Stat`): Spectrum statistics container for normalization
        """
        super(Model, self).__init__()

        self.resample_ratio = resample_ratio

        # Speaker-independent Encoder: segFC-Conv-LSTM // segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM
        self.encoder = Taco2Encoder(conf.encoder)

        # Global speaker conditioning network
        self.global_cond = GlobalCondNet(conf.global_cond)

        # Decoder
        ## PreNet: (segFC-ReLU-DO)xN
        self.register_buffer("target_mean", torch.from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", torch.from_numpy(stats.scale_).float())
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

    def forward(self, features, lens, spk_emb, targets = None):
        """Convert unit sequence into acoustic feature sequence.

        Args:
            features (Batch, T_max, Feature_i): input unit sequences
            lens
            spk_emb (Batch, Spk_emb): speaker embedding vectors as global conditioning
            targets (Batch, T_max, Feature_o): padded target acoustic feature sequences
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
