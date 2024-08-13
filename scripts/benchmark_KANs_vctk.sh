#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

ids=(225 234 238 245 248 253 335 345 363 374)
seqs=(023)
mic=1

pes=(None)

activations=(
    fourier
    bspline
)

# NeRF 270 K
# FFN 280 K
# None 263 K
base_path=data/VCTK/wav48_silence_trimmed


for id in "${ids[@]}"
do
    for seq_id in "${seqs[@]}"
    do
        data_path=$base_path/p${id}/p${id}_${seq_id}_mic${mic}.flac
        for nonlin in "${activations[@]}"
        do
            save_dir=logs/benchmark_KANs_VCTK/p${id}_${seq_id}_mic${mic}/$nonlin
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
done
