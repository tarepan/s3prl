# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2q/expert.py ]
#   Synopsis     [ the wav2vec2q wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan & Tarepan, Japan]
"""*********************************************************************************************"""

"""wav2vec2 with `q` output"""


import argparse
from typing import List
from packaging import version

import torch
import fairseq
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from s3prl.utility.helper import zero_mean_unit_var_norm


class UpstreamExpert(UpstreamBase):
    """wav2vec 2.0 model with q output.

    Wrapper of Fairseq `Wav2Vec2Model`.
    `self.model.quantizer` are forward-hooked,
    so the quantized representative vector series can be extracted.
    """

    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        # Validation: Fairseq version
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        # Read only single model (length of `[ckpt]` is 1), so access the model.
        self.model = model[0] # ::Wav2Vec2Model
        self.wav_normalize = cfg.task.normalize

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False

        # Add forward hooks
        # upstream_output includes (unique_identifier, transformed_IO) =
        #     ("self.model.quantizer": forward_output["x"])
        if len(self.hooks) == 0:
            self.add_hook("self.model.quantizer", lambda input, output: output["x"])

    def get_downsample_rates(self, key: str) -> int:
        """Yield downsampling rate of wav2vec 2.0

        wav2vec 2.0 encoder use strided convs, so downsampled.
        It is common between variants, and the value is 5*2*2*2*2*2*2=320.
        """
        return 320

    def forward(self, wavs):
        """Encode waveforms into quantized feature sequences q.

        Args:
            wavs:
        Returns:
            Void (handled in UpstreamBase by forward hook of nn.Module)
        """
        device = wavs[0].device

        # Input normalization
        if self.wav_normalize:
            if self.numpy_wav_normalize:
                wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
                wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
            else:
                wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        # Input padding
        padded_wav = pad_sequence(wavs, batch_first=True)

        # Forward calculation
        results = self.model.quantize(padded_wav)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
