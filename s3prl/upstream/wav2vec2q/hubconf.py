# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2q/hubconf.py ]
#   Synopsis     [ the wav2vec 2.0 q torch hubconf ]
#   Author       [ S3PRL / Kushal Lakhotia & Tarepan]
"""*********************************************************************************************"""


"""wav2vec 2.0 variants with q output.

- wav2vec2_q (wav2vec2_q_large_960)
- wav2vec2_q_large_960
- wav2vec2_q_large_ll60k
- wav2vec2_q_xlsr
"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
# `_` is not exported by `hub` module
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_q_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_q_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_q_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec2_q(refresh=False, *args, **kwargs):
    """
        The default model w/ q output - Base
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_q_base_960(refresh=refresh, *args, **kwargs)


def wav2vec2_q_base_960(refresh=False, *args, **kwargs):
    """
        The Base model w/ q output
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt'
    return wav2vec2_q_url(refresh=refresh, *args, **kwargs)


def wav2vec2_q_large_960(refresh=False, *args, **kwargs):
    """
        The Large model w/ q output trained on LibriSpeech 960 hours of data
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'
    return wav2vec2_q_url(refresh=refresh, *args, **kwargs)    


def wav2vec2_q_large_ll60k(refresh=False, *args, **kwargs):
    """
        The Large model w/ q output trained on Libri-light 60k hours of data
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt'
    return wav2vec2_q_url(refresh=refresh, *args, **kwargs)


def wav2vec2_q_xlsr(refresh=False, *args, **kwargs):
    """
        The wav2vec 2.0 model w/ q output trained on multilingual presented in https://arxiv.org/abs/2006.13979
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt'
    return wav2vec2_q_url(refresh=refresh, *args, **kwargs)
