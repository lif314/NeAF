#!/bin/bash

# Positional encodings are parameter sensitive

export CUDA_VISIBLE_DEVICES=0

datatypes=(blues00000)
pes=(FFN)
# activations=(
#     relu
#     gaussian
#     sine
#     learnable-sine
# )

activations=(
    relu
)

# orders=(2 4 8 16 32 64 128 256)
orders=(256)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_FFN_L/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for L in "${orders[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --exp_name "${nonlin}_${L}" \
                    --batch_size 16384 \
                    --mapping_input $L \
                    --audio_path $data_path
            done
        done
    done
done
