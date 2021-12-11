# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ The VCTK + VCC2020 dataset ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import os
import random

import librosa
import numpy as np
from tqdm import tqdm
from speechcorpusy import load_preset
from speechdatasety.helper.adress import dataset_adress, generate_path_getter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from resemblyzer import preprocess_wav, VoiceEncoder
from .utils import logmelspectrogram
from .utils import read_hdf5, write_hdf5


# Hardcoded VCC2020 speaker names for evaluation
SRCSPKS = ["SEF1", "SEF2", "SEM1", "SEM2"]
TRGSPKS_TASK1 = ["TEF1", "TEF2", "TEM1", "TEM2"]

# Hardcoded resampling rate
FS = 16000


def generate_eval_pairs(file_list, train_file_list, eval_data_root, num_samples):
    """Generate VCC2020 adress pairs for evaluation.

    All (source_utterance, target_speaker) pairs are generated.
    target_speaker utterances are random sampled from a speaker.
    Args:
        file_list: Source file list
        train_file_list: Target file list
        eval_data_root: Root adress of evaluation data
        num_samples: Number of evaluation target per source
    Returns:
        (List[[adress]]): #0 is source, #1~ are target
    """
    # X::Pair[]
    X = []
    for trgspk in TRGSPKS_TASK1:

        # Target .wav file adress list of a speaker, filtered out those exist
        spk_file_list = []
        for number in train_file_list:
            # eval_data_root / trgspk / f"{number}.wav"
            wav_path = os.path.join(eval_data_root, trgspk, number + ".wav")
            if os.path.isfile(wav_path):
                spk_file_list.append(wav_path)

        # generate pairs
        for srcspk in SRCSPKS:
            for number in file_list:
                random.shuffle(spk_file_list)
                # Pair: [adress_src1_1, adress_tgt1_5, adress_tgt1_3, ..., adress_tgt_num_samples]
                # eval_data_root / srcspk / f"{number}.wav"
                pair = [os.path.join(eval_data_root, srcspk, number + ".wav")]
                pair.extend(spk_file_list[:num_samples])
                X.append(pair)
    return X


class VCTK_VCC2020Dataset(Dataset):
    """dataset for a2a-vc-vctk task

    Training:   VCTK
    Evaluation: VCC2020
    """

    def __init__(self, split, 
                 trdev_data_root, eval_data_root, spk_embs_root, 
                 lists_root, eval_lists_root,
                 fbank_config, spk_emb_source, num_ref_samples,
                 train_dev_seed=1337, **kwargs):
        """
        Prepare .wav paths, then generate speaker embedding if needed

        Data split: train/dev/test = [:-11, -5:, -10:-5] for each speaker

        Args:
            trdev_data_root: Root adress of train/dev corpus (VCTK)
            download: Whether to download train/dev corpus if needed
            eval_data_root: Root adress of wav file for evaluation (VCC2020)
            lists_root: Root adress of utterance list
            eval_lists_root: Root adress of evaluation utterance list
            fbank_config: Filterback configurations
            spk_emb_source ("external" | Any): Flag of embedding
            train_dev_seed: Random seed, affect item order
        """
        super(VCTK_VCC2020Dataset, self).__init__()
        self.split = split
        self.fbank_config = fbank_config
        self.spk_emb_source = spk_emb_source
        self.spk_embs_root = spk_embs_root
        os.makedirs(spk_embs_root, exist_ok=True)

        # Directories
        ## Test
        ### Step1
        # f"{lists_root}/eval_{num_samples}sample_list.txt"
        # f"{eval_lists_root}/eval_list.txt"
        # f"{eval_lists_root}/E_train_list.txt"
        # f"{eval_data_root}/{trgspk}/{number}.wav"
        # f"{eval_data_root}/{srcspk}/{number}.wav"
        ### Step2
        # f"{self.spk_embs_root}/TE{J}{N}_E{N}00{NN}.h5
        # {self.spk_embs_root}/TE{J}{N}_E{N}00{NN}.h5
        #     (inHDF5)/spk_emb

        # Design Notes:
        #   HDF5:
        #     HDF5 is over-enginerring.
        #     Currently it is used only for embeddings, and all embeddings are saved
        #     in different .h5 files.
        #     No dataset-wide access, no partial access.

        # Step1 of `X`: Prepare .wav paths as material
        X = []
        ## Train/dev: [ItemID]
        if split == 'train' or split == 'dev':
            self.corpus = load_preset("VCTK", root=trdev_data_root, download=download)
            self.corpus.get_contents()

            adress_archive, self._path_contents = dataset_adress(
                trdev_data_root,
                self.corpus.__class__.__name__,
                "emb",
                "hashed_args",
            )

            # Prepare datum path getter.
            ## (.h5 => .pt)
            ## f"{self._path_contents}/{spk}/embs/p{NNN}_{NNN}.emb.pt"
            self.get_path_emb = generate_path_getter("emb", self._path_contents)

            # Prepare data identities.
            ## In each speakers, [0, -10] is for train, [-5:] is for dev
            all_utterances = self.corpus.get_identities()
            is_train = split == 'train'
            for spk in set(map(lambda item_id: item_id.speaker, all_utterances)):
                utts_spk = filter(lambda item_id: item_id.speaker == spk, all_utterances)
                X.extend(utts_spk[:-10] if is_train else utts_spk[-5:])
            random.seed(train_dev_seed)
            random.shuffle(X)

        ## Test: [wav_source_path, wav_target_1_path, wav_target_2_path, ...][]
        elif split == 'test':
            for num_samples in num_ref_samples:
                # lists_root / f"eval_{num_samples}sample_list.txt"
                eval_pair_list_file = os.path.join(lists_root, "eval_{}sample_list.txt".format(num_samples))
                # Load saved pathes
                if os.path.isfile(eval_pair_list_file):
                    print("[Dataset] eval pair list file exists: {}".format(eval_pair_list_file))
                    with open(eval_pair_list_file, "r") as f:
                        lines = f.read().splitlines()
                    X += [line.split(",") for line in lines]
                # Generate pathes
                else:
                    print("[Dataset] eval pair list file does not exist: {}".format(eval_pair_list_file))
                    # generate eval pairs
                    ## eval_lists_root / "eval_list.txt"
                    with open(os.path.join(eval_lists_root, 'eval_list.txt')) as f:
                        file_list = f.read().splitlines()
                    ## eval_lists_root / "E_train_list.txt"
                    with open(os.path.join(eval_lists_root, 'E_train_list.txt')) as f:
                        train_file_list = f.read().splitlines()
                    # Evaluation adress pairs
                    eval_pairs = generate_eval_pairs(file_list, train_file_list, eval_data_root, num_samples)
                    # write in file
                    with open(eval_pair_list_file, "w") as f:
                        for line in eval_pairs:
                            f.write(",".join(line)+"\n")
                    X += eval_pairs
        else:
            raise ValueError('Invalid \'split\' argument for dataset: VCTK_VCC2020Dataset!')
        print('[Dataset] - number of data for ' + split + ': ' + str(len(X)))
        self.X = X
        # /Prepare .wav adress

        # Be careful, `self.X` could be updated by `extract_spk_embs()`
        if spk_emb_source == "external":
            # extract spk embs beforehand
            print("[Dataset] Extracting speaker emebddings")
            self.extract_spk_embs()
        else:
            NotImplementedError


    def _extract_a_spk_emb(self, wav_path, spk_emb_path, spk_encoder):
        """Extract speaker embedding from an untterance."""
        ## on-memory preprocessing
        wav = preprocess_wav(wav_path)
        embedding = spk_encoder.embed_utterance(wav)
        # save spk emb
        # spk_emb_path/(inHDF5)spk_emb
        write_hdf5(str(spk_emb_path), "spk_emb", embedding.astype(np.float32))

    def extract_spk_embs(self):
        """"Update path object, then generate and save embedding"""
        # load speaker encoder
        spk_encoder = VoiceEncoder()

        # Step2 of `X`: set wav and embedding
        ## Train/dev: [wav, self_embedding]
        if self.split == "train" or self.split == "dev":
            ### Backward compatibility
            new_X = []
            for item_id in tqdm(self.X, desc="Extracting speaker embedding"):
                wav_path = self.corpus.get_item_path(item_id)
                spk_emb_path = self.get_path_emb(item_id)
                if not spk_emb_path.is_file():
                    self._extract_a_spk_emb(wav_path, spk_emb_path, spk_encoder):
                ### backward compatibility
                new_X.append([wav_path, spk_emb_path])
            ### backward compatibility
            self.X = new_X
        # Test: [src_wav, tgt_emb_1, tgt_emb_2, ...]
        elif self.split == "test":
            new_X = []
            for wav_paths in self.X:
                source_wav_path = wav_paths[0]
                # :[wav_path, emb_1_path, emb_2_path, ...]
                new_tuple = [source_wav_path]
                # Embeddings are needed only for target speaker because this is VC task.
                for wav_path in wav_paths[1:]:
                    spk, number = wav_path.split(os.sep)[-2:]
                    # f"{self.spk_embs_root}/TE{J}{N}_E{N}00{NN}.h5
                    spk_emb_path = os.path.join(self.spk_embs_root, spk + "_" + number.replace(".wav", ".h5"))
                    new_tuple.append(spk_emb_path)
                    if not os.path.isfile(spk_emb_path):
                        # {self.spk_embs_root}/TE{J}{N}_E{N}00{NN}.h5/(inHDF5)spk_emb = embedding
                        self._extract_a_spk_emb(wav_path, spk_emb_path, spk_encoder):
                new_X.append(new_tuple)
            # ::[wav_path, emb_1_path, emb_2_path, ...][]
            self.X = new_X

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
        return len(self.X)

    def get_all_lmspcs(self):
        """Acquire log-mel spectrograms from all waveforms.

        Returns:
            [(Time, MelFreq)] List of log-mel spectrogram
        """

        lmspcs = []
        for xs in tqdm(self.X, dynamic_ncols=True, desc="Extracting target acoustic features"):
            input_wav_path = xs[0]
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
            input_wav_resample (ndarray): The waveform, could be resampled by `FS`
            input_wav_original (ndarray): The waveform, acquired with sr=fbank_config["fs"]
            lmspc: log-mel spectrogram
            ref_spk_emb: Averaged self|target speaker embedding
            input_wav_path: Path of .wav file, modified when split==`test`
            ref_spk_name: Speaker name of embedding
        """
        input_wav_path = self.X[index][0]
        # train/dev: 1 self embedding
        # test: multiple target embedding
        spk_emb_paths = self.X[index][1:]
        # Speaker name of embedding
        ref_spk_name = os.path.basename(spk_emb_paths[0]).split("_")[0]

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

        # get speaker embeddings
        if self.spk_emb_source == "external":
            ref_spk_embs = [read_hdf5(spk_emb_path, "spk_emb") for spk_emb_path in spk_emb_paths]
            ref_spk_embs = np.stack(ref_spk_embs, axis=0)
            ref_spk_emb = np.mean(ref_spk_embs, axis=0)
        else:
            ref_spk_emb = None

        # Test split: change input wav path name
        if self.split == "test":
            input_wav_name = input_wav_path.replace(".wav", "")
            input_wav_path = input_wav_name + "_{}samples.wav".format(len(spk_emb_paths))

        return input_wav_resample, input_wav_original, lmspc, ref_spk_emb, input_wav_path, ref_spk_name
    
    def collate_fn(self, batch):
        """collate function used by dataloader.

        Sort data with feature time length, then pad features.
        Args:
            batch: (B, input_wav_resample, input_wav_original, lmspc, ref_spk_emb, input_wav_path, ref_spk_name)
        Returns:
            wavs: List[Tensor(`input_wav_resample`)]
            wavs_2: List[Tensor(`input_wav_original`)]
            acoustic_features: List[Tensor(`lmspc`)]
            acoustic_features_padded: `acoustic_features` padded by PyTorch function
            acoustic_feature_lengths: Tensor(feature time length)
            wav_paths: List[`input_wav_path`]
            ref_spk_embs: Tensor(`ref_spk_emb`)
            ref_spk_names: List[`ref_spk_name`]
        """

        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])

        bs = len(sorted_batch) # batch_size
        wavs = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        wavs_2 = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)] # This is used for obj eval
        acoustic_features = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        acoustic_features_padded = pad_sequence(acoustic_features, batch_first=True)
        acoustic_feature_lengths = torch.from_numpy(np.array([acoustic_feature.size(0) for acoustic_feature in acoustic_features]))
        ref_spk_embs = torch.from_numpy(np.array([sorted_batch[i][3] for i in range(bs)]))
        wav_paths = [sorted_batch[i][4] for i in range(bs)]
        ref_spk_names = [sorted_batch[i][5] for i in range(bs)]
        
        return wavs, wavs_2, acoustic_features, acoustic_features_padded, acoustic_feature_lengths, wav_paths, ref_spk_embs, ref_spk_names
