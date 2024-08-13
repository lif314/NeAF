#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

datatypes=(bach counting blues00000)
pes=(NeRF FFN None)

activations=(
    relu
    prelu
    selu
    tanh
    sigmoid
    silu
    softplus
    elu
    sinc
    quadratic
    multi-quadratic
    gaussian
    laplacian
    super-gaussian
    expsin
    gabor-wavelet
    sine
    learnable-sine
)

for type in "${datatypes[@]}"
do
    data_path=data/demo/gt_$type.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_MLPs_demo/$type/$nonlin
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
