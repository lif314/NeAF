#!/bin/bash

# Positional encodings are parameter sensitive

export CUDA_VISIBLE_DEVICES=0

# datatypes=(bach counting blues00000)
datatypes=(blues00000)
pes=(FFN)
activations=(
    gaussian
    sine
    learnable-sine
)

# scales=(1 20 40 60 80 100 1000 10000)
scales=(1000 10000)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_FFN_sigma/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for scale in "${scales[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --exp_name "${nonlin}_${scale}" \
                    --batch_size 16384 \
                    --ffn_scale $scale \
                    --audio_path $data_path
            done
        done
    done
done
