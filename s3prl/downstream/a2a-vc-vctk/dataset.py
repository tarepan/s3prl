# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ The VCTK + VCC2020 dataset ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import os
import random
from typing import List
from pathlib import Path
import pickle
from dataclasses import dataclass

import librosa
import numpy as np
from tqdm import tqdm
from speechcorpusy import load_preset
from speechdatasety.helper.archive import hash_args
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive
from speechdatasety.helper.adress import dataset_adress, generate_path_getter
from speechdatasety.interface.speechcorpusy import ItemId

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from resemblyzer import preprocess_wav, VoiceEncoder
from .utils import logmelspectrogram
from .utils import read_hdf5, write_hdf5


# Hardcoded resampling rate for upstream
FS = 16000


def split_jvs(utterances: List[ItemId]) -> (List[ItemId], List[ItemId]):
    """Split JVS corpus items into two groups."""

    anothers_spk = ["95", "96", "98", "99"]
    # Filter for train/test split of single corpus
    ones = list(filter(
        lambda item_id: item_id.speaker not in anothers_spk,
        utterances
    ))
    anothers = list(filter(
        lambda item_id: item_id.speaker in anothers_spk,
        utterances
    ))
    return ones, anothers


@dataclass
class Stat:
    """Spectrogarm statistics container"""
    mean_: np.ndarray
    scale_: np.ndarray


def save_vc_tuples(content_path: Path, num_target: int, tuples: List[List[ItemId]]):
    p = content_path / f"vc_{num_target}_tuples.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(tuples, f)

def load_vc_tuples(content_path: Path, num_target: int) -> List[List[ItemId]]:
    p = content_path / f"vc_{num_target}_tuples.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    else:
        Exception(f"{str(p)} does not exist.")


def generate_vc_tuples(
    sources: List[ItemId],
    targets: List[ItemId],
    num_target: int,
    ) -> List[ItemId]:
    """Generate utterance tuples for voice convertion.

    VC needs a source utterance (content source) and multiple target utterances (style target).
    This function generate this tuples for all source utterances.
    Target utterances are randomly sampled from utterances of single target speaker.
    Args:
        sources: Source utterances
        targets: Target utterances
        num_target: Target utterances (that will be averaged in downstream) per single source
    Returns:
        (List[ItemId]): #0 is a source, #1~ are targets
    """
    target_speakers = set(map(lambda item_id: item_id.speaker, targets))
    full_list = []
    for trgspk in target_speakers:
        # "all_sources to trgspk" vc tuple
        utts_of_trgspk = list(filter(lambda item_id: item_id.speaker == trgspk, targets))
        for src in sources:
            vc_tuple = [src, *random.sample(utts_of_trgspk, num_target)]
            full_list.append(vc_tuple)
    return full_list


class VCTK_VCC2020Dataset(Dataset):
    """dataset for a2a-vc-vctk task

    Training:   VCTK
    Evaluation: VCC2020
    """

    def __init__(self,
        split: str,
        adress_data_root: str,
        fbank_config,
        num_target: int,
        corpus_train_dev: str,
        corpus_test: str,
        num_dev_sample: int,
        download: bool = False,
        train_dev_seed=1337,
        **kwargs
        ):
        """
        Prepare voice conversion tuples (source-targets tuple), then generate speaker embedding.

        Data split: [train, dev] = [:-11, -5:, -10:-5] for each speaker

        Args:
            split: "train" | "dev" | "test"
            adress_data_root: Root adress of train/dev corpus (VCTK)
            fbank_config: Filterback configurations
            num_target: Number of target utterances per single source utterance
            corpus_train_dev: Name of corpus for train/dev
            corpus_test: Name of corpus for test
            num_dev_sample: Number of dev samples per single speaker
            download: Whether to download train/dev corpus if needed
            train_dev_seed: Random seed, affect item order
        """
        super().__init__()
        self.split = split
        self.fbank_config = fbank_config
        self._num_target = num_target

        # Design Notes:
        #   HDF5:
        #     HDF5 is over-enginerring.
        #     Currently it is used only for embeddings, and all embeddings are saved
        #     in different .h5 files.
        #     No dataset-wide access, no partial access.

        # Get corpus
        corpus_name = corpus_train_dev if (split == 'train' or split == 'dev') else corpus_test
        self._corpus = load_preset(corpus_name, root=adress_data_root, download=download)

        # Extract corpus contents
        # This dataset do not save raw waves in dataset archive, so always need corpus contents.
        self._corpus.get_contents()

        # Construct dataset adresses
        adress_archive, self._path_contents = dataset_adress(
            adress_data_root,
            self._corpus.__class__.__name__,
            "emb",
            f"{split}_hashed_args",
        )
        self.get_path_emb = generate_path_getter("emb", self._path_contents)
        self._path_stats = self._path_contents / "stats.pkl"

        # Select data identities.
        all_utterances = self._corpus.get_identities()
        ## list of [content_source_utt, style_target_utt_1, style_target_utt_2, ...]
        self._vc_tuples: List[List[ItemId]] = []
        ## list of style target, which will be preprocessed as embedding
        self._targets: List[ItemId] = []

        if split == 'train' or split == 'dev':
            if corpus_name == "JVS":
                all_utterances = split_jvs(all_utterances)[0]
            # target is self, source:target = 1:1
            ## Data split: [0, -2X] is for train, [-X:] is for dev for each speaker
            is_train = split == 'train'
            idx_dev = -1*num_dev_sample
            for spk in set(map(lambda item_id: item_id.speaker, all_utterances)):
                utts_spk = list(map(
                    # self target
                    lambda item_id: [item_id, item_id],
                    filter(lambda item_id: item_id.speaker == spk, all_utterances)
                ))
                self._vc_tuples.extend(utts_spk[:2*idx_dev] if is_train else utts_spk[idx_dev:])
            random.seed(train_dev_seed)
            random.shuffle(self._vc_tuples)
            self._targets = list(map(lambda vc_tuple: vc_tuple[1], self._vc_tuples))

        elif split == 'test':
            # target is other speaker, source:target = 1:N
            if corpus_name == "VCC20":
                # Missing utterances in original code: E10001-E10050 (c.f. tarepan/s3prl#2)
                self._sources = list(filter(lambda item_id: item_id.subtype == "eval_source", all_utterances))
                self._targets = list(filter(lambda i: i.subtype == "train_target_task1", all_utterances))
            elif corpus_name == "JVS":
                all_utterances = split_jvs(all_utterances)[1]
                # 10 utterances per speaker for test source
                self._sources = []
                for spk in set(map(lambda item_id: item_id.speaker, all_utterances)):
                    utts_spk = list(filter(lambda item_id: item_id.speaker == spk, all_utterances))
                    self._sources.extend(utts_spk[:10])
                # All test utterances are target style
                self._targets = all_utterances
            else:
                Exception(f"Corpus '{corpus_name}' is not yet supported for test split")

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")

        # Load vc tuples in the file
        if split == 'test':
            self._vc_tuples = load_vc_tuples(self._path_contents, num_target)

        # Report
        print(f"[Dataset] - number of data for {split}: {len(self._vc_tuples)}")

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        # Embedding
        spk_encoder = VoiceEncoder()
        for item_id in tqdm(self._targets, desc="Preprocessing", unit="utterance"):
            self._extract_a_spk_emb(
                self._corpus.get_item_path(item_id),
                self.get_path_emb(item_id),
                spk_encoder,
            )

        # Statistics
        if self.split == "train":
            self._calculate_spec_stat()

        # VC tuples
        if self.split == "test":
            # Generate vc tuples randomly
            vc_tuples = generate_vc_tuples(self._sources, self._targets, self._num_target)
            save_vc_tuples(self._path_contents, self._num_target, vc_tuples)

    def _extract_a_spk_emb(self, wav_path, spk_emb_path, spk_encoder):
        """Extract speaker embedding from an untterance."""
        ## on-memory preprocessing
        wav = preprocess_wav(wav_path)
        embedding = spk_encoder.embed_utterance(wav)
        # save spk emb
        # spk_emb_path/(inHDF5)spk_emb
        write_hdf5(str(spk_emb_path), "spk_emb", embedding.astype(np.float32))

    def acquire_spec_stat(self):
        """Acquire scaler, the statistics (mean and variance) of mel-spectrograms"""
        with open(self._path_stats, "rb") as f:
            scaler =  pickle.load(f)
        return scaler

    def _calculate_spec_stat(self):
        """Calculate mean and variance of spectrograms."""
        
        # ※ Need high memory because extract all specs at once
        # [(Time, MelFreq)]
        lmspcs = self.get_all_lmspcs()

        ## Calculate average
        # ave::(MelFreq)
        ave = np.zeros(lmspcs[0].shape[1])
        L = 0
        # spectrogram::(Time, MelFreq)
        for spectrogram in lmspcs:
            ave = np.add(ave, np.sum(spectrogram, axis=0))
            L += spectrogram.shape[0]
        ave = ave/L

        ## Calculate sigma
        # sigma::(MelFreq)
        sigma = np.zeros(lmspcs[0].shape[1])
        L = 0
        # spectrogram::(Time, MelFreq)
        for spectrogram in lmspcs:
            sigma = np.add(sigma, np.sum(np.abs(spectrogram - ave), axis=0))
            L += spectrogram.shape[0]
        sigma = sigma/L

        scaler = Stat(ave, sigma)

        # Save
        with open(self._path_stats, "wb") as f:
            pickle.dump(scaler, f)

    def _load_wav(self, wav_path, fs):
        """Load wav file with resampling if needed

        Args:
            wav_path: Adress of waveform
            fs: Sampling frequency
        Returns:
            (Time) Single-channel waveform
        """
        # use librosa to resample. librosa gives range [-1, 1]
        # mono=True in default, so this is single-channel waveform
        wav, sr = librosa.load(wav_path, sr=fs)
        return wav, sr

    def __len__(self):
        """Number of .wav files (and same number of embeddings)"""
        return len(self._vc_tuples)

    def get_all_lmspcs(self):
        """Acquire log-mel spectrograms from all waveforms.

        Returns:
            [(Time, MelFreq)] List of log-mel spectrogram
        """

        lmspcs = []
        for xs in tqdm(self._vc_tuples, dynamic_ncols=True, desc="Extracting target acoustic features"):
            # input_wav_path = 
            input_wav_path = self._corpus.get_item_path(xs[0])
            # (Time), (Time)
            input_wav_original, fs_original = self._load_wav(input_wav_path, fs=None)
            # (Time) => (Time, MelFreq)
            lmspc = logmelspectrogram(
                x=input_wav_original,
                fs=fs_original,
                n_mels=self.fbank_config["n_mels"],
                n_fft=self.fbank_config["n_fft"],
                n_shift=self.fbank_config["n_shift"],
                win_length=self.fbank_config["win_length"],
                window=self.fbank_config["window"],
                fmin=self.fbank_config["fmin"],
                fmax=self.fbank_config["fmax"],
            )
            lmspcs.append(lmspc)
        return lmspcs
        

    def __getitem__(self, index):
        """
        Load raw waveform, resampled waveform and log-mel-spec in place (not preprocessed).

        Returns:
            input_wav_resample (ndarray): Waveform used by Upstream (should be sr=FS)
            None
            lmspc: log-mel spectrogram
            ref_spk_emb: Averaged self|target speaker embedding
            input_wav_path: Path of .wav file, modified when split==`test`
            ref_spk_name: Speaker name of embedding
        """

        selected = self._vc_tuples[index]
        input_wav_path = self._corpus.get_item_path(selected[0])
        spk_emb_paths = list(map(lambda item_id: self.get_path_emb(item_id), selected[1:]))
        # Speaker name of target embedding
        ref_spk_name = selected[1].speaker

        # FS: Target sampling rate (global variable)
        input_wav_original, _ = self._load_wav(input_wav_path, fs=self.fbank_config["fs"])
        input_wav_resample, fs_resample = self._load_wav(input_wav_path, fs=FS)

        # ad-hoc spectrogram generation
        lmspc = logmelspectrogram(
            x=input_wav_original,
            fs=self.fbank_config["fs"],
            n_mels=self.fbank_config["n_mels"],
            n_fft=self.fbank_config["n_fft"],
            n_shift=self.fbank_config["n_shift"],
            win_length=self.fbank_config["win_length"],
            window=self.fbank_config["window"],
            fmin=self.fbank_config["fmin"],
            fmax=self.fbank_config["fmax"],
        )

        # An averaged embedding of the speaker's N utterances
        ref_spk_embs = [read_hdf5(spk_emb_path, "spk_emb") for spk_emb_path in spk_emb_paths]
        ref_spk_embs = np.stack(ref_spk_embs, axis=0)
        ref_spk_emb = np.mean(ref_spk_embs, axis=0)

        # Test split: change input wav path name
        if self.split == "test":
            input_wav_name = str(input_wav_path).replace(".wav", "")
            input_wav_path = f"{input_wav_name}_{len(spk_emb_paths)}samples.wav"

        return input_wav_resample, None, lmspc, ref_spk_emb, str(input_wav_path), ref_spk_name
    
    def collate_fn(self, batch):
        """collate function used by dataloader.

        Sort data with feature time length, then pad features.
        Args:
            batch: (B, input_wav_resample, None, lmspc, ref_spk_emb, input_wav_path, ref_spk_name)
        Returns:
            wavs: List[Tensor(`input_wav_resample`)]
            acoustic_features: List[Tensor(`lmspc`)]
            acoustic_features_padded: `acoustic_features` padded by PyTorch function
            acoustic_feature_lengths: Tensor(feature time length)
            wav_paths: List[`input_wav_path`]
            ref_spk_embs: Tensor(`ref_spk_emb`)
            ref_spk_names: List[`ref_spk_name`]
        """

        sorted_batch = sorted(batch, key=lambda x: -x[0].shape[0])

        bs = len(sorted_batch) # batch_size
        wavs = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        acoustic_features = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        acoustic_features_padded = pad_sequence(acoustic_features, batch_first=True)
        acoustic_feature_lengths = torch.from_numpy(np.array([acoustic_feature.size(0) for acoustic_feature in acoustic_features]))
        ref_spk_embs = torch.from_numpy(np.array([sorted_batch[i][3] for i in range(bs)]))
        wav_paths = [sorted_batch[i][4] for i in range(bs)]
        ref_spk_names = [sorted_batch[i][5] for i in range(bs)]
        
        return wavs, acoustic_features, acoustic_features_padded, acoustic_feature_lengths, wav_paths, ref_spk_embs, ref_spk_names
