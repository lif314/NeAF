#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

datatypes=(bach counting blues00000)
pes=(NeRF FFN None)

activations=(
    fourier
)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benchmark_pe_kans/$type/$nonlin
        for pe in "${pes[@]}"
        do
            python train.py --arch $nonlin \
                --pe_type $pe \
                --save_dir $save_dir \
                --input_grid_size 10 \
                --hidden_grid_size 10 \
                --output_grid_size 10 \
                --hidden_layers 3 \
                --hidden_features 64 \
                --exp_name "${nonlin}_${pe}" \
                --batch_size 16384 \
                --audio_path $data_path
        done
    done
done
