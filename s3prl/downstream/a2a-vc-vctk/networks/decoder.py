"""Network modules of Decoder"""


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfDecoderPreNet:
    """Configuration of Decoder's PreNet

    Args:
        idim - Dimension size of input
        n_layers - Number of FC layer
        n_units - Dimension size of hidden and final layers
        dropout_rate: float=0.5    
    """
    idim: int
    n_layers: int
    n_units: int
    dropout_rate: float

class Taco2Prenet(torch.nn.Module):
    """Prenet module for decoder of Tacotron2.

    Model: (FC-ReLU-DO)xN | DO

    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps alleviate the exposure bias problem.

    Note:
        This module alway applies dropout even in evaluation.
        See the detail in `Natural TTS Synthesis by
        Conditioning WaveNet on Mel Spectrogram Predictions`_.

    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, conf: ConfDecoderPreNet):
        """Initialize"""
        super(Taco2Prenet, self).__init__()
        self._conf = conf
        self.dropout_rate = conf.dropout_rate
        # WholeNet
        self.prenet = torch.nn.ModuleList()
        for layer in range(conf.n_layers):
            n_inputs = conf.idim if layer == 0 else conf.n_units
            self.prenet += [
                # FC-ReLU
                torch.nn.Sequential(torch.nn.Linear(n_inputs, conf.n_units), torch.nn.ReLU())
            ]

    def forward(self, x):
        # Make sure at least one dropout is applied even when there is no FC layer.
        if len(self.prenet) == 0:
            return F.dropout(x, self.dropout_rate)

        for i in range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x