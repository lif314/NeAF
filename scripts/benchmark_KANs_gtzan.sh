#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

datatypes=(blues classical country disco hiphop jazz metal pop reggae rock)
seq_id=00000

# pes=(NeRF FFN None)
pes=(None)

activations=(
    fourier
    bspline
)

# NeRF 270 K
# FFN 280 K
# None 263 K
base_path=data/gtzan/genres

for type in "${datatypes[@]}"
do
    data_path=$base_path/$type/$type.$seq_id.wav
    for nonlin in "${activations[@]}"
    do
        save_dir=logs/benckmark_kan_gtzan/$type/$nonlin
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
