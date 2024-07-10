#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# datatypes=(bach counting)
datatypes=(counting)

# archs=(incode wire siren fourier)
# archs=(wire siren fourier)
archs=(wire fourier)

for type in "${datatypes[@]}"
do
    data_path=data/siren/gt_$type.wav
    for nonlin in "${archs[@]}"
    do
        save_dir=logs/benckmark_nopes/$type
        python train.py --arch $nonlin \
            --pe_type None \
            --save_dir $save_dir \
            --exp_name $nonlin \
            --batch_size 16384 \
            --audio_path $data_path
    done
done
