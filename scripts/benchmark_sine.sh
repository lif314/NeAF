#!/bin/bash

# Activation functions are parameter sensitive

export CUDA_VISIBLE_DEVICES=0

# datatypes=(bach counting blues00000)
datatypes=(bach counting)
pes=(NeRF None)
activations=(
    sine
)

# default omega 300
# 3 30 300 3000 30000 300000
sine_as=(0.01 0.1 1.0 10.0 100.0 1000.0)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_sine/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for a in "${sine_as[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --a $a \
                    --exp_name "${nonlin}_${a}" \
                    --batch_size 16384 \
                    --audio_path $data_path
            done
        done
    done
done
