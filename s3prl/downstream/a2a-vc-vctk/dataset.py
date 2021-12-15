# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ The VCTK + VCC2020 dataset ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import random
from typing import List
from pathlib import Path
import pickle
from dataclasses import dataclass

from scipy.io import wavfile
import librosa
import soundfile as sf
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
from .utils import logmelspectrogram, read_npy, write_npy


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

        corpus_name = corpus_train_dev if (split == 'train' or split == 'dev') else corpus_test
        self._corpus = load_preset(corpus_name, root=adress_data_root, download=download)

        # Construct dataset adresses
        adress_archive, self._path_contents = dataset_adress(
            adress_data_root,
            self._corpus.__class__.__name__,
            "wav_emb_mel_vctuple",
            f"{split}_{num_dev_sample}forDev_{num_target}targets",
        )
        self._get_path_wav = generate_path_getter("wav", self._path_contents)
        self._get_path_emb = generate_path_getter("emb", self._path_contents)
        self._get_path_mel = generate_path_getter("mel", self._path_contents)
        self._path_stats = self._path_contents / "stats.pkl"

        # Select data identities.
        all_utterances = self._corpus.get_identities()
        ## list of [content_source_utt, style_target_utt_1, style_target_utt_2, ...]
        self._vc_tuples: List[List[ItemId]] = []
        ## list of content source, which will be preprocessed as resampled waveform
        self._sources: List[ItemId] = []
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
            self._sources = list(map(lambda vc_tuple: vc_tuple[0], self._vc_tuples))
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

        self._corpus.get_contents()

        # Waveform resampling for upstream input
        # Low sampling rate is enough because waveforms are finally encoded into compressed feature.
        for item_id in tqdm(self._sources, desc="Preprocess/Resampling", unit="utterance"):
            wave, _ = librosa.load(self._corpus.get_item_path(item_id), sr=FS)
            p = self._get_path_wav(item_id)
            p.parent.mkdir(exist_ok=True, parents=True)
            sf.write(p, wave, FS, format="WAV")

        # Embedding
        spk_encoder = VoiceEncoder()
        for item_id in tqdm(self._targets, desc="Preprocess/Embedding", unit="utterance"):
            wav = preprocess_wav(self._corpus.get_item_path(item_id))
            embedding = spk_encoder.embed_utterance(wav)
            write_npy(self._get_path_emb(item_id), embedding.astype(np.float32))

        # Mel-spectrogram
        for item_id in tqdm(self._sources, desc="Preprocess/Melspectrogram", unit="utterance"):
            # 'Not too downsampled' waveform for feature generation
            wave, sr = librosa.load(self._corpus.get_item_path(item_id), sr=self.fbank_config["fs"])
            lmspc = logmelspectrogram(
                x=wave,
                fs=sr,
                n_mels=self.fbank_config["n_mels"],
                n_fft=self.fbank_config["n_fft"],
                n_shift=self.fbank_config["n_shift"],
                win_length=self.fbank_config["win_length"],
                window=self.fbank_config["window"],
                fmin=self.fbank_config["fmin"],
                fmax=self.fbank_config["fmax"],
            )
            write_npy(self._get_path_mel(item_id), lmspc)

        # Statistics
        if self.split == "train":
            self._calculate_spec_stat()
            print("Preprocess/Stats (only in `train`) - done")

        # VC tuples
        if self.split == "test":
            # Generate vc tuples randomly
            vc_tuples = generate_vc_tuples(self._sources, self._targets, self._num_target)
            save_vc_tuples(self._path_contents, self._num_target, vc_tuples)
            print("Preprocess/VC_tuple (only in `test`) - done")

    def acquire_spec_stat(self):
        """Acquire scaler, the statistics (mean and variance) of mel-spectrograms"""
        with open(self._path_stats, "rb") as f:
            scaler =  pickle.load(f)
        return scaler

    def _calculate_spec_stat(self):
        """Calculate mean and variance of source spectrograms."""

        # Implementation Notes:
        #   Dataset could be huge, so loading all spec could cause memory overflow.
        #   For this reason, this implementation repeat 'load a spec and stack stats'.

        # average spectrum over source utterances :: (MelFreq)
        spec_stack = None
        L = 0
        for item_id in self._sources:
            # lmspc::(Time, MelFreq)
            lmspc = read_npy(self._get_path_mel(item_id))
            uttr_sum = np.sum(lmspc, axis=0)
            spec_stack = np.add(spec_stack, uttr_sum) if spec_stack is not None else uttr_sum
            L += lmspc.shape[0]
        ave = spec_stack/L

        ## sigma in each frequency bin :: (MelFreq)
        sigma_stack = None
        L = 0
        for item_id in self._sources:
            # lmspc::(Time, MelFreq)
            lmspc = read_npy(self._get_path_mel(item_id))
            uttr_sigma_sum = np.sum(np.abs(lmspc - ave), axis=0)
            sigma_stack = np.add(sigma_stack, uttr_sigma_sum) if sigma_stack is not None else uttr_sigma_sum
            L += lmspc.shape[0]
        sigma = sigma_stack/L

        scaler = Stat(ave, sigma)

        # Save
        with open(self._path_stats, "wb") as f:
            pickle.dump(scaler, f)

    def __len__(self):
        """Number of .wav files (and same number of embeddings)"""
        return len(self._vc_tuples)

    def __getitem__(self, index):
        """Load waveforms, mel-specs, speaker embeddings and data identities.

        Returns:
            input_wav_resample (ndarray): Waveform used by Upstream (should be sr=FS)
            lmspc (ndarray): log-mel spectrogram
            ref_spk_emb: Averaged self|target speaker embeddings
            vc_identity (str, str, str): (target_speaker, source_speaker, utterance_name)
        """

        selected = self._vc_tuples[index]
        source_id = selected[0]
        target_ids = selected[1:]

        # Preprocessing is done w/ `librosa`, so no worry of details (mono, bit-depth, etc).
        _, input_wav_resample = wavfile.read(self._get_path_wav(source_id))

        lmspc = read_npy(self._get_path_mel(source_id))

        # An averaged embedding of the speaker's N utterances
        ref_spk_embs = [read_npy(self._get_path_emb(item_id)) for item_id in target_ids]
        ref_spk_emb = np.mean(np.stack(ref_spk_embs, axis=0), axis=0)

        # VC identity (target_speaker,        source_speaker,    utterance_name)
        vc_identity = (target_ids[0].speaker, source_id.speaker, source_id.name)

        return input_wav_resample, lmspc, ref_spk_emb, vc_identity
    
    def collate_fn(self, batch):
        """collate function used by dataloader.

        Sort data with feature time length, then pad features.
        Args:
            batch: (B, input_wav_resample, lmspc, ref_spk_emb, vc_identity)
        Returns:
            wavs: List[Tensor(`input_wav_resample`)]
            acoustic_features: List[Tensor(`lmspc`)]
            acoustic_features_padded: `acoustic_features` padded by PyTorch function
            acoustic_feature_lengths: Tensor(feature time length)
            ref_spk_embs: Tensor(`ref_spk_emb`)
            vc_ids: List[(target_speaker, source_speaker, utterance_name)]
        """

        # Sort
        sorted_batch = sorted(batch, key=lambda item: -item[0].shape[0])

        wavs =                  list(map(lambda item: torch.from_numpy(item[0]), sorted_batch))
        acoustic_features =     list(map(lambda item: torch.from_numpy(item[1]), sorted_batch))
        acoustic_features_padded = pad_sequence(acoustic_features, batch_first=True)
        acoustic_feature_lengths = torch.from_numpy(np.array(list(map(lambda feat: feat.size(0), acoustic_features))))
        ref_spk_embs = torch.from_numpy(np.array(list(map(lambda item: item[2],  sorted_batch))))
        vc_ids =               list(map(lambda item:                   item[3],  sorted_batch))

        return wavs, acoustic_features, acoustic_features_padded, acoustic_feature_lengths, ref_spk_embs, vc_ids
