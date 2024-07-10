# type=blues
# seq_id=00000

type=$1
seq_id=$2

base_path=data/gtzan/genres
data_path=$base_path/$type/$type.$seq_id.wav
save_dir=logs/gtzan/$type/$seq_id

# # 254 K
# python train.py --arch fourier \
#     --save_dir $save_dir \
#     --exp_name fourier \
#     --batch_size 16384 \
#     --audio_path $data_path


# # 263 K
# python train.py --arch siren \
#     --save_dir $save_dir \
#     --exp_name siren \
#     --batch_size 16384 \
#     --audio_path $data_path

# 263 K
python train.py --arch relu \
    --save_dir $save_dir \
    --exp_name relu \
    --batch_size 16384 \
    --audio_path $data_path
