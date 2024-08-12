#!/bin/bash

# Activation functions are parameter sensitive

export CUDA_VISIBLE_DEVICES=0

# datatypes=(bach counting blues00000)
datatypes=(bach counting)
pes=(NeRF FFN None)
activations=(
    gaussian
)

# gaussian_as=(0.1 1.0 5.0 10.0 20.0 50.0 100.0)
gaussian_as=(0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_gaussian/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for a in "${gaussian_as[@]}"
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
