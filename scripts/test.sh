type=blues
seq_id=00000

# type=$1
# seq_id=$2

base_path=data/gtzan/genres
data_path=$base_path/$type/$type.$seq_id.wav
save_dir=logs/test_mlp/$type/$seq_id


# # test init
# python train.py --arch fourier \
#     --save_dir $save_dir \
#     --init_type uniform \
#     --exp_name fourier_uniform_no_first_out \
#     --batch_size 16384 \
#     --outermost_linear False \
#     --audio_path $data_path

# python train.py --arch fourier \
#     --save_dir $save_dir \
#     --init_type norm \
#     --exp_name fourier_norm_no_first_out \
#     --batch_size 16384 \
#     --outermost_linear False \
#     --audio_path $data_path

# Test mlp

# 270 K
python train.py --arch relu \
    --save_dir $save_dir \
    --exp_name relu_gaussian100 \
    --pe_type FFN \
    --batch_size 16384 \
    --audio_path $data_path

# python train.py --arch relu \
#     --save_dir $save_dir \
#     --exp_name relu_FFN \
#     --pe_type FFN \
#     --batch_size 16384 \
#     --audio_path $data_path

# python train.py --arch relu \
#     --save_dir $save_dir \
#     --exp_name relu_None \
#     --pe_type None \
#     --batch_size 16384 \
#     --audio_path $data_path

# python train.py --arch fourier \
#     --save_dir $save_dir \
#     --exp_name fourier_NeRF \
#     --pe_type NeRF \
#     --input_grid_size 32 \
#     --hidden_grid_size 5 \
#     --output_grid_size 5 \
#     --batch_size 16384 \
#     --audio_path $data_path

# python train.py --arch fourier \
#     --save_dir $save_dir \
#     --init_type rand \
#     --exp_name fourier_rand_out \
#     --batch_size 16384 \
#     --outermost_linear False \
#     --audio_path $data_path

# Test Outlinear
# 254 K
# python train.py --arch hyper \
#     --save_dir $save_dir \
#     --exp_name fourier \
#     --input_grid_size 1024 \
#     --hidden_layers 4 \
#     --hidden_features 64 \
#     --hidden_grid_size 3 \
#     --output_grid_size 3 \
#     --batch_size 16384 \
#     --outermost_linear False \
#     --audio_path $data_path


# 274 K
# python train.py --arch hyper \
#     --save_dir $save_dir \
#     --input_grid_size 1024 \
#     --hidden_grid_size 3 \
#     --output_grid_size 3 \
#     --hidden_layers 5 \
#     --hidden_features 64 \
#     --exp_name hyper \
#     --batch_size 8192 \
#     --audio_path $data_path


# 263 K
# python train.py --arch siren \
#     --save_dir $save_dir \
#     --exp_name siren \
#     --batch_size 16384 \
#     --num_epochs 3000 \
#     --audio_path $data_path
