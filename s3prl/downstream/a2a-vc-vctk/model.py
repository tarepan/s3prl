# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the simple (LSTMP), simple-AR and taco2-AR models for any-to-any voice conversion ]
#   Reference    [ `WaveNet Vocoder with Limited Training Data for Voice Conversion`, Interspeech 2018 ]
#   Reference    [ `Non-Parallel Voice Conversion with Autoregressive Conversion Model and Duration Adjustment`, Joint WS for BC & VCC 2020 ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""

"""Basically same with A2O model, but embedding is added. 'Added for A2A' are annotation of the differences."""


from warnings import warn
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from extorch import Conv1dEx

from .dataset import Stat
from .networks.encoder import Taco2Encoder, ConfEncoder
from .networks.conditioning import GlobalCondNet, ConfGlobalCondNet
from .networks.decoder import Taco2Prenet, ConfDecoderPreNet

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling
    
    Model: 1RNN[-LN][-DO][-DownSample][-FC-Tanh]
    '''

    def __init__(self, input_dim, module: str, bidirection, dim, dropout, layer_norm: bool, sample_rate, proj: bool):
        """
        Args:
            input_dim
            module: module name for dynamic RNN type switching ...? (e.g."LSTM")
            bidirection
            dim
            dropout: Dropout probability
            layer_norm: Whether to use LayerNormalization
            sample_rate: downsample if >1 else None
            proj: Whether to use non-linear projection
        """
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj

        # RNN: 1RNN
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Projection: FC-Tanh
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):

        # RNN
        if not self.training:
            self.layer.flatten_parameters()
        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)

        # Dropout
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            # Now not defined anywhere...?
            output, x_len = downsample(output, x_len, self.sample_rate, 'drop')

        # Projection
        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class RNNCell(nn.Module):
    ''' RNN cell wrapper'''

    def __init__(self, input_dim, module, dim, dropout:float, layer_norm:bool, proj:bool):
        super(RNNCell, self).__init__()
        # Setup
        rnn_out_dim = dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.proj = proj

        # RNN cell
        self.cell = getattr(nn, module.upper()+"Cell")(input_dim, dim)

        # Normalization
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)

        # Dropout
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Projection: FC-Tanh
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)
    
    def forward(self, input_x, z, c):

        new_z, new_c = self.cell(input_x, (z, c))

        if self.layer_norm:
            new_z = self.ln(new_z)

        if self.dropout > 0:
            new_z = self.dp(new_z)

        if self.proj:
            new_z = torch.tanh(self.pj(new_z))

        return new_z, new_c

################################################################################

class ConfModel:
    """Configuration of Model"""
    input_dim: int,        # Dimension size of input feature
    output_dim: int,       # Dimension size of output mel-spectrum
    hidden_dim: int,
    resample_ratio: float, # Time-directional up/down sampling ratio toward input series
    # Encoder
    encoder_type: str,     # "taco2" | "ffn"
    enc_bidi: bool,        # Whether to use bidirectional
    enc_conv_causal: bool, # Whether to use causal convolution
    # Decoder
    ## PreNet
    prenet_layers: int,         # Number of layers
    prenet_dim: int,            # Number of each layer's dimension
    prenet_dropout_rate: float, # Dropout rate
    ## MainNet
    ar: bool,                  # Whether teacher-forcing or not
    lstmp_layers: int,         # Number of layers
    lstmp_dropout_rate: float, # Dropout rate
    lstmp_layernorm: bool,     # Whether to use LayerNorm
    stats: Stat,               # Spectrum normalization stats
    # 'Added for A2A'
    spk_emb_integration_type: str, # "add" | "concat"
    spk_emb_dim: int,              # Number of embedding dimension

class Model(nn.Module):
    """
    S3PRL-VC model.

    `Taco2-AR`: segFC-3Conv-1LSTM-cat_(z_t, AR-segFC)-NuniLSTM-segFC-segLinear
    segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM-cat_(z_t, AR-norm-(segFC-ReLU-DO)xN)-(1uniLSTM[-LN][-DO]-segFC-Tanh)xL-segFC
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 resample_ratio,
                 stats,
                 ar,
                 encoder_type,
                 hidden_dim,
                 lstmp_layers: int,
                 lstmp_dropout_rate,
                 lstmp_proj_dim,
                 lstmp_layernorm: bool,
                 # 'Added for A2A'
                 spk_emb_integration_type,
                 spk_emb_dim,
                 # /
                 prenet_layers=2,
                 prenet_dim=256,
                 prenet_dropout_rate=0.5,
                 enc_bidi: bool = True,
                 enc_conv_causal: bool = False,
                 **kwargs):
        """
        Args:
            stats (`Stat`): Statistics object for normalization
            enc_bidi: Whether Encoder LSTM is bidirectional or not. 
        """
        super(Model, self).__init__()

        self.ar = ar
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim # this is also the decoder output dim
        self.output_dim = output_dim # acoustic feature dim
        self.resample_ratio = resample_ratio

        # Encoder
        ## `Taco2-AR`: segFC-Conv-biLSTM
        if encoder_type == "taco2":
            # model: segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1LSTM`h`, `h` stand for `hidden_dim`
            self.encoder = Taco2Encoder(
                ConfEncoder(
                    input_dim, eunits=hidden_dim, bidirectional=enc_bidi, causal=enc_conv_causal
            ))
        ## `simple` | `simple-AR`: segFC-ReLU
        elif encoder_type == "ffn":
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU()
            )
        else:
            raise ValueError("Encoder type not supported.")

        # Speaker conditioning
        self.cond_net = GlobalCondNet(
            ConfGlobalCondNet(spk_emb_integration_type, hidden_dim, spk_emb_dim)
        )

        # Decoder
        ## Normalization
        self.register_buffer("target_mean", torch.from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", torch.from_numpy(stats.scale_).float())
        ## PreNet, `Taco2-AR` only: (segFC-ReLU-DO)xN
        self.prenet = Taco2Prenet(
            ConfDecoderPreNet(
                idim=output_dim,
                n_layers=prenet_layers,
                n_units=prenet_dim,
                dropout_rate=prenet_dropout_rate,
        ))
        ## MainNet: LSTMP + linear projection
        ### lstmps: (1uniLSTM[-LN][-DO]-segFC-Tanh)xN
        self.lstmps = nn.ModuleList()
        for i in range(lstmp_layers):
            if ar:
                prev_dim = output_dim if prenet_layers == 0 else prenet_dim
                rnn_input_dim = hidden_dim + prev_dim if i == 0 else hidden_dim
                rnn_layer = RNNCell(
                    rnn_input_dim,
                    "LSTM",
                    hidden_dim,
                    lstmp_dropout_rate,
                    lstmp_layernorm,
                    proj=True,
                )
            else:
                rnn_input_dim = hidden_dim
                rnn_layer = RNNLayer(
                    rnn_input_dim,
                    "LSTM",
                    False, # bidirectional
                    hidden_dim,
                    lstmp_dropout_rate,
                    lstmp_layernorm,
                    sample_rate=1,
                    proj=True,
                )
            self.lstmps.append(rnn_layer)
        ### Projection: segFC
        self.proj = torch.nn.Linear(hidden_dim, output_dim)
        ## PostNet: None
        pass

    def normalize(self, x):
        """Normalize spectrum for AR"""
        return (x - self.target_mean) / self.target_scale
    
    def forward(self, features, lens, spk_emb, targets = None):
        """Convert unit sequence into acoustic feature sequence.

        Args:
            features (Batch, T_max, Feature_i): input unit sequences
            lens
            spk_emb (Batch, Spk_emb): speaker embedding vectors as global conditioning
            targets (Batch, T_max, Feature_o): padded target acoustic feature sequences
        """
        B = features.shape[0]

        # Resampling: resample the input features according to resample_ratio
        # (B, T_max, Feat_i) => (B, Feat_i, T_max) => (B, Feat_i, T_max') => (B, T_max', Feat_i)
        features = features.permute(0, 2, 1)
        resampled_features = F.interpolate(features, scale_factor = self.resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self.resample_ratio

        # Encoder :: (resampled_features:(B, T_max', Feat_i)) -> (B, T_max', Feat_h)
        ## `Taco2-AR`
        if self.encoder_type == "taco2":
            encoder_states, lens = self.encoder(resampled_features, lens)
        ## `simple` | `simple-AR`
        elif self.encoder_type == "ffn":
            encoder_states = self.encoder(resampled_features)

        # Global speaker conditioning
        encoder_states = self.cond_net(encoder_states, spk_emb)

        # Decoder: spec_t' = f(spec_t, cond_t), cond_t == f(unit_t, spk_g)
        # AR decofing w/ or w/o teacher-forcing
        if self.ar:
            # Transpose for easy access: (B, T_max, Feat_o) => (T_max, B, Feat_o)
            if targets is not None:
                targets = targets.transpose(0, 1)
            predicted_list = []

            # Initialize LSTM hidden state and cell state of all LSTMP layers, and x_t-1
            c_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            z_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            for _ in range(1, len(self.lstmps)):
                c_list += [encoder_states.new_zeros(B, self.hidden_dim)]
                z_list += [encoder_states.new_zeros(B, self.hidden_dim)]
            prev_out = encoder_states.new_zeros(B, self.output_dim)

            # step-by-step loop for autoregressive decoding
            ## encoder_state::(B, hidden_dim)
            for t, encoder_state in enumerate(encoder_states.transpose(0, 1)):
                # Single time step
                ## PreNet(t-1) & Concat
                concat = torch.cat([encoder_state, self.prenet(prev_out)], dim=1) # each encoder_state has shape (B, hidden_dim)
                ## Run single time step of all LSTMP layers
                for i, lstmp in enumerate(self.lstmps):
                    # Run a layer (1uniLSTM[-LN][-DO]-segFC-Tanh), then update states
                    # Input: (latent_t, t-1) OR below layer's hidden state
                    lstmp_input = concat if i == 0 else z_list[i-1]
                    z_list[i], c_list[i] = lstmp(lstmp_input, z_list[i], c_list[i])
                # Projection & Stack: Stack output_t `proj(o_lstmps)` in full-time list
                predicted_list += [self.proj(z_list[-1]).view(B, self.output_dim, -1)] # projection is done here to ensure output dim
                # teacher-forcing if `target` else pure-autoregressive
                prev_out = targets[t] if targets is not None else predicted_list[-1].squeeze(-1)
                # AR spectrum is normalized (todo: could be moved up, but it change t=0 behavior)
                prev_out = self.normalize(prev_out)
                # /Single time step
            predicted = torch.cat(predicted_list, dim=2)
            predicted = predicted.transpose(1, 2)  # (B, hidden_dim, Lmax) -> (B, Lmax, hidden_dim)
        else:
            predicted = encoder_states
            for i, lstmp in enumerate(self.lstmps):
                predicted, lens = lstmp(predicted, lens)

            # projection layer
            predicted = self.proj(predicted)

        return predicted, lens
