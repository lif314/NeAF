#!/bin/bash

# Activation functions are parameter sensitive

export CUDA_VISIBLE_DEVICES=0

datatypes=(bach counting blues00000)
pes=(None)
activations=(
    siren
)

first_omegas=(3 30 300 3000 30000 300000)
hidden_omegas=(30)

for type in "${datatypes[@]}"
do
    data_path=data/demo/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_siren/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for hidden_omega in "${hidden_omegas[@]}"
            do
                for first_omega in "${first_omegas[@]}"
                do
                    python train.py --arch $nonlin \
                        --pe_type $pe \
                        --save_dir $save_dir \
                        --first_omega_0 $first_omega \
                        --hidden_omega_0 $hidden_omega \
                        --exp_name "${nonlin}_${first_omega}_${hidden_omega}" \
                        --batch_size 16384 \
                        --audio_path $data_path
                done
            done
        done
    done
done
