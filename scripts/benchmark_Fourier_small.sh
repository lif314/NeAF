#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# datatypes=(bach counting blues00000)
datatypes=(bach)
# pes=(NeRF FFN None)
pes=(None)

activations=(
    fourier
)

# 1024 5 3
# 1: 172K 15.10
# 2: 213K 28.20
# 3: 254K 
# hidden_layers=(1 2)
hidden_layers=(3)

# 1024 8 3
# 1: 197K 17.20
# 2: 262K 35.10
# 3: 254K 

for type in "${datatypes[@]}"
do
    data_path=data/demo/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benchmark_fourier_small/$type/$nonlin
        for pe in "${pes[@]}"
        do
            for hidden_layer in "${hidden_layers[@]}"
            do
                python train.py --arch $nonlin \
                    --pe_type $pe \
                    --save_dir $save_dir \
                    --input_grid_size 10 \
                    --hidden_grid_size 5 \
                    --output_grid_size 3 \
                    --hidden_layers $hidden_layer \
                    --hidden_features 64 \
                    --exp_name "${nonlin}_${pe}_${hidden_layer}" \
                    --batch_size 16384 \
                    --audio_path $data_path
            done
        done
    done
done