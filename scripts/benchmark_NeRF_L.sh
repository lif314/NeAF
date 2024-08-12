#!/bin/bash

# Positional encodings are parameter sensitive

export CUDA_VISIBLE_DEVICES=1

datatypes=(bach counting blues00000)
pes=(NeRF)
activations=(
    relu
    gaussian
    sine
    learnable-sine
)

orders=(2 4 8 16 32 64)
# Params: 264 265 268 272 280 296

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_NeRF_test/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for L in "${orders[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --exp_name "${nonlin}_${L}" \
                    --batch_size 16384 \
                    --num_frequencies $L \
                    --audio_path $data_path
            done
        done
    done
done
