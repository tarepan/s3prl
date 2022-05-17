#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ vocoder_download.sh ]
#   Synopsis     [ Script to download pretrained neural vocoders ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

# This script is based on the following links:
# https://raw.githubusercontent.com/espnet/espnet/master/egs/vcc20/vc1_task1/local/pretrained_model_download.sh
# https://github.com/espnet/espnet/blob/master/utils/download_from_google_drive.sh

# Directory under which checkpoint archive will be expanded
download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

# id: 12w1LpF6HjsJBmOUUkS6LV1d7AX18SA7u
# It seems to be HiFi-GAN checkpoint provided by S3PRL-VC...? We cannot find it in Kan-bayashi PWG repository.
hifigan_vctk_url="https://drive.google.com/open?id=12w1LpF6HjsJBmOUUkS6LV1d7AX18SA7u"

download_from_google_drive() {
    share_url=$1 # ${hifigan_vctk_url}
    dir=$2       # ${download_dir}/hifigan_vctk
    file_ext=$3  # ".tar.gz"

    # make temp dir
    [ ! -e "${dir}" ] && mkdir -p "${dir}"
    tmp=$(mktemp "${dir}/XXXXXX.${file_ext}")

    # download & decompress
    file_id=$(echo "${share_url}" | cut -d"=" -f 2) # 12w1LpF6HjsJBmOUUkS6LV1d7AX18SA7u
    gdown --id "${file_id}" -O "${tmp}"
    tar xvzf "${tmp}" -C "${dir}" # Extract archive contents into `${download_dir}/hifigan_vctk`

    # remove tmp
    rm "${tmp}"
}

download_from_google_drive ${hifigan_vctk_url} ${download_dir}/hifigan_vctk ".tar.gz"
echo "Successfully finished donwload of pretrained models."
