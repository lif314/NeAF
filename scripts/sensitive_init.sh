#!/bin/bash

# Positional encodings are parameter sensitive

export CUDA_VISIBLE_DEVICES=1

datatypes=(bach counting)
pes=(None NeRF FFN)
activations=(
    sine_xavier
    sine_normal
    sine
)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_init/$type/$nonlin
        for pe in "${pes[@]}"
        do
            python train.py --arch $nonlin \
                --pe_type $pe \
                --save_dir $save_dir \
                --exp_name "${nonlin}" \
                --batch_size 16384 \
                --audio_path $data_path
        done
    done
done
