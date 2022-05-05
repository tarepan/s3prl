"""Network modules of Encoder"""


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from extorch import Conv1dEx


# The follow section is related to Tacotron2
# Reference: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


@dataclass
class ConfEncoder:
    """
    Args:
        idim: Dimension of the inputs.
        elayers: The number of encoder LSTM layers.
        eunits:  The number of encoder LSTM units.
        econv_layers: The number of encoder conv layers.
        econv_filts:  The number of encoder conv filter size.
        econv_chans:  The number of encoder conv filter channels.
        use_batch_norm: Whether to use batch normalization.
        use_residual: Whether to use residual connection.
        dropout_rate: Dropout rate.
        bidirectional: Whether LSTM is bidirectional or not.
        causal: Whether Conv1d is causal or not.
    """
    idim: int
    elayers: int = 1
    eunits: int = 512
    econv_layers: int = 3
    econv_chans: int = 512
    econv_filts: int = 5
    use_batch_norm: bool = True
    use_residual: bool = False
    dropout_rate: float = 0.5
    bidirectional: bool = True
    causal: bool = False
    
class Taco2Encoder(torch.nn.Module):
    """Encoder module of the Tacotron2 TTS model.

    Model: segFC[-(Res(Conv1d[-BN]-ReLU-DO))xN][-MbiLSTM]

    Reference:
    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, conf: ConfEncoder):
        """Initialize Tacotron2 encoder module.

        By default, model is segFC512-(Conv1d512_k5s1-BN-ReLU-DO_0.5)x3-1biLSTM512
        """
        super(Taco2Encoder, self).__init__()
        # store the hyperparameters
        self._conf = conf
        self.idim = conf.idim
        self.use_residual = conf.use_residual

        # segFC linear
        self.input_layer = torch.nn.Linear(conf.idim, conf.econv_chans)

        # convs: [(Conv1d[-BN]-ReLU-DO)xN]
        if conf.econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(conf.econv_layers):
                ichans = conf.econv_chans
                # BN on/off, remaining is totally same
                ## BN (+)
                if conf.use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            Conv1dEx(
                                ichans,
                                conf.econv_chans,
                                conf.econv_filts,
                                stride=1,
                                padding=(conf.econv_filts - 1) // 2,
                                bias=False,
                                causal=conf.causal,
                            ),
                            torch.nn.BatchNorm1d(conf.econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(conf.dropout_rate),
                        )
                    ]
                ## BN (-)
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            Conv1dEx(
                                ichans,
                                conf.econv_chans,
                                conf.econv_filts,
                                stride=1,
                                padding=(conf.econv_filts - 1) // 2,
                                bias=False,
                                causal=conf.causal,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(conf.dropout_rate),
                        )
                    ]
        else:
            self.convs = None

        # blstm: [N-LSTM]
        if conf.elayers > 0:
            iunits = conf.econv_chans if conf.econv_layers != 0 else embed_dim
            dim_lstm = conf.eunits // 2 if conf.bidirectional else conf.eunits
            self.blstm = torch.nn.LSTM(
                iunits, dim_lstm, conf.elayers, batch_first=True, bidirectional=conf.bidirectional
            )
            print(f"Encoder LSTM: {'bidi' if conf.bidirectional else 'uni'}")
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.
        Args:
            xs (Batch, T_max, Feature_o): padded acoustic feature sequence
        """

        # segFC linear
        xs = self.input_layer(xs).transpose(1, 2)

        # Conv
        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)

        # LSTM
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Lmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens