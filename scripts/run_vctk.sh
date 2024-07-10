export CUDA_VISIBLE_DEVICES=1

id=225
seq=001
mic=1  # 1 OR 2

# type=$1
# seq_id=$2

base_path=data/VCTK/wav48_silence_trimmed
data_path=$base_path/p${id}/p${id}_${seq}_mic${mic}.wav
save_dir=logs/VCTK/p${id}_${seq}_mic${mic}

# 254 K
python train.py --arch fourier \
    --save_dir $save_dir \
    --exp_name fourier \
    --batch_size 16384 \
    --audio_path $data_path


# 263 K
python train.py --arch siren \
    --save_dir $save_dir \
    --exp_name siren \
    --batch_size 16384 \
    --audio_path $data_path
