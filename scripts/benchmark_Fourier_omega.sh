#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

datatypes=(bach counting blues00000)
# pes=(NeRF FFN None)
pes=(None)

activations=(
    fourier
)

# 1024 512  256  128  64   32   16   8    5
# 254K 189K 156K 139K 131K 127K 125K 124K 124K
input_omega=(1024 512 256 128 64 32 16 8 5)

for type in "${datatypes[@]}"
do
    data_path=data/demo/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benchmark_fourier_omega/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for omega in "${input_omega[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --input_grid_size $omega \
                    --hidden_grid_size 5 \
                    --output_grid_size 3 \
                    --hidden_layers 3 \
                    --hidden_features 64 \
                    --exp_name "${nonlin}_${pe}_${omega}" \
                    --batch_size 16384 \
                    --audio_path $data_path
            done
        done
    done
done