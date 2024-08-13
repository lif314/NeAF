#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

datatypes=(bach counting)
pes=(None NeRF FFN)
activations=(
    sine_xavier
    sine_normal
    sine
)

for type in "${datatypes[@]}"
do
    data_path=data/demo/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_sensitive_init/$type/$nonlin
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
