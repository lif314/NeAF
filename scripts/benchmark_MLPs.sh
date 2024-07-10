#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

datatypes=(bach counting)
pes=(NeRF FFN None)
# activations=(
#     relu
#     prelu
#     selu
#     tanh
#     sigmoid
#     silu
#     softplus
#     elu
#     sinc
#     gaussian
#     quadratic
#     multi-quadratic
#     laplacian
#     super-gaussian
#     expsin
# )
activations=(
    selu
    tanh
    sigmoid
    silu
    softplus
    elu
    sinc
    gaussian
    quadratic
    multi-quadratic
    laplacian
    super-gaussian
    expsin
)

# NeRF 270 K
# FFN 280 K
# None 263 K

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_mlp/$type/$nonlin
        for pe in "${pes[@]}"
        do
            python train.py --arch $nonlin \
                --pe_type $pe \
                --save_dir $save_dir \
                --exp_name "${nonlin}_${pe}" \
                --batch_size 16384 \
                --audio_path $data_path
        done
    done
done
