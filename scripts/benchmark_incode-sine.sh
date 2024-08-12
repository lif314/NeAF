#!/bin/bash

# Activation functions are parameter sensitive

export CUDA_VISIBLE_DEVICES=0

# datatypes=(bach counting blues00000)
datatypes=(bach counting)
pes=(NeRF FFN None)
activations=(
    learnable-sine
)

sine_as=(3 30 300 3000 30000 300000)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_incode-sine/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for a in "${sine_as[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --hidden_omega_0 $a \
                    --exp_name "${nonlin}_${a}" \
                    --batch_size 16384 \
                    --audio_path $data_path
            done
        done
    done
done
