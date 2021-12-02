<div align="center">

# Intra-lingual A2A VC / S3PRL-VC <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.2110.06280-B31B1B.svg)][paper]

</div>

Intra-lingual Any-to-Any Voice Conversion based on S3PRL; S3PRL-VC.  

## Task
Intra-lingual A2A/any-to-any voice conversion.  
Trained on **VCTK**, evaluated on **VCC2020**.  
More specifically, evaluation method is same as VCC2020 Task1.  

## Implementation

- model:
  - wave2mel: any S3PRL upstreams
  - unit2mel: **Taco2-AR**
    - speaker embedding: [d-vector] implementation by the [Resemblyzer]
  - mel2wave: **[HiFi-GAN]**, [kan-bayashi's implementation][HiFi-GAN_impl]

[d-vector]: https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41939.pdf
[Resemblyzer]: https://github.com/resemble-ai/Resemblyzer
[HiFi-GAN]: https://arxiv.org/abs/2010.05646
[HiFi-GAN_impl]: https://github.com/kan-bayashi/ParallelWaveGAN

## Dependencies:

- `parallel-wavegan`
- `fastdtw`
- `pyworld`
- `pysptk`
- `jiwer`
- `resemblyzer`

You can install them via the `requirements.txt` file.

## Usage

### Preparation
```
# Download the VCTK and the VCC2020 datasets.
cd <root-to-s3prl>/s3prl/downstream/a2a-vc-vctk
cd data
./vcc2020_download.sh vcc2020/
./vctk_download.sh ./
cd ../

# Download the pretrained vocoders.
./vocoder_download.sh ./
```

### Training 
The following command starts a training run given any `<upstream>`.
```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/vc_train.sh <upstream> downstream/a2a-vc-vctk/config_ar_taco2.yaml <tag>
```
Along the training process, you may find converted speech samples generated using the Griffin-Lim algorithm automatically saved in `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_<tag>_<upstream>/<step>/test/wav/`.
**NOTE**: to avoid extracting d-vectors on-the-fly (which is very slow), all d-vectors are extracted beforehand and saved in `data/spk_embs`. Since there are 44 hours of data in VCTK, the whole extraction can take a long time. On a NVIDIA GeForce RTX 3090, it takes 5-6 hours.
**NOTE 2**: By default, during testing, the d-vector of the target speaker is the average of random samples from the training set, of number `num_ref_samples`. You can change this number in the config file. The list of samples is generated automatically and saved in `data/eval_<num>sample_list.txt`.

### Waveform synthesis (decoding) using a neural vocoder & objective evaluation

#### Single model checkpoint decoding & evaluation
```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/decode.sh <vocoder> <result_dir>/<step>
```
For example,
```
./downstream/a2a-vc-vctk/decode.sh ./downstream/a2a-vc-vctk/hifigan_vctk result/downstream/a2a_vc_vctk_taco2_ar_decoar2/50000
```

#### Upstream-wise decoding & evaluation
The following command performs objective evaluation of a model trained with a specific number of steps.
```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/batch_vc_decode.sh <upstream> taco2_ar downstream/a2a-vc-vctk/hifigan_vctk
```
If the command fails, please make sure there are trained results in `result/downstream/a2a_vc_vctk_<tag>_<upstream>/`. The generated speech samples will be saved in `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_taco2_ar_<upstream>/<step>/hifigan_wav/`. 

Also, the output of the evaluation will be shown directly:
```
decoar2 10 samples epoch 48000 best: 9.28 41.80 0.197 1.3 4.0 27.00
```
And detailed utterance-wise evaluation results can be found in `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_taco2_ar_<upstream>/<step>/hifigan_wav/obj_10samples.log`.

## Related Tasks
**A2O/any-to-one** recipe: [a2o-vc-vcc2020](../a2o-vc-vcc2020/)

## Citation

If you find this recipe useful, please consider citing following paper:
```
@article{huang2021s3prl,
  title={S3PRL-VC: Open-source Voice Conversion Framework with Self-supervised Speech Representations},
  author={Huang, Wen-Chin and Yang, Shu-Wen and Hayashi, Tomoki and Lee, Hung-Yi and Watanabe, Shinji and Toda, Tomoki},
  journal={arXiv preprint arXiv:2110.06280},
  year={2021}
}
```

## Contact
Development: [Wen-Chin Huang](https://github.com/unilight) @ Nagoya University (2021).  
If you have any questions, please open an issue, or contact through email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp


[paper]: https://arxiv.org/abs/2110.06280
[notebook]: https://colab.research.google.com/github/tarepan/s3prl/blob/master/s3prl/downstream/a2a-vc-vctk/training.ipynb