#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

datatypes=(bach counting blues00000)
pes=(NeRF FFN None)

activations=(
    bspline
)

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
                --input_grid_size 32 \
                --hidden_grid_size 5 \
                --output_grid_size 3 \
                --hidden_layers 4 \
                --hidden_features 64 \
                --exp_name "${nonlin}_${pe}" \
                --batch_size 16384 \
                --audio_path $data_path
        done
    done
done
