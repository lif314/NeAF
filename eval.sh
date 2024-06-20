# export CUDA_VISIBLE_DEVICES=1

data_path="data/siren/gt_bach.wav"
# data_path="data/siren/gt_counting.wav"
# data_path="data/gtzan/genres/blues/blues.00000.wav"

# 198K
python train.py --arch siren \
    --hidden_layers 3 \
    --hidden_features 256 \
    --wav_path $data_path \
    --save_dir logs/Ebach \
    --exp_name siren


# 263 K
python train.py --arch siren \
    --wav_path $data_path \
    --hidden_layers 4 \
    --hidden_features 256 \
    --save_dir logs/Ebach \
    --exp_name siren_4_256

# 326 K
python train.py --arch siren \
    --wav_path $data_path \
    --hidden_layers 5 \
    --hidden_features 256 \
    --save_dir logs/Ebach \
    --exp_name siren_5_256

# 395 K
python train.py --arch siren \
    --wav_path $data_path \
    --hidden_layers 6 \
    --hidden_features 256 \
    --save_dir logs/Ebach \
    --exp_name siren_6_256


# 189 K
# python train.py --arch fourier --wav_path $data_path --save_dir logs/blues00 --exp_name fourier

# 139 K
python train.py --arch fourier \
    --wav_path $data_path \
    --input_grid_size 512 \
    --hidden_grid_size 3 \
    --output_grid_size 3 \
    --save_dir logs/Ebach \
    --exp_name fourier_512_33

# 205 K
python train.py --arch fourier \
    --wav_path $data_path \
    --input_grid_size 1024 \
    --hidden_grid_size 3 \
    --output_grid_size 3 \
    --save_dir logs/Ebach \
    --exp_name fourier_1024_33

# 254K
python train.py --arch fourier \
    --wav_path $data_path \
    --input_grid_size 1024 \
    --hidden_grid_size 5 \
    --output_grid_size 3 \
    --save_dir logs/Ebach \
    --exp_name fourier_1024_53

# 598 K
python train.py --arch fourier \
    --wav_path $data_path \
    --input_grid_size 4096 \
    --hidden_grid_size 3 \
    --output_grid_size 3 \
    --save_dir logs/Ebach \
    --exp_name fourier_4096_33


# 574 K
python train.py --arch fourier \
    --wav_path $data_path \
    --hidden_layers 2 \
    --hidden_features 64 \
    --input_grid_size 4096 \
    --hidden_grid_size 3 \
    --output_grid_size 3 \
    --save_dir logs/Ebach \
    --exp_name fourier2_64_4096_33

# 526 K
python train.py --arch fourier \
    --wav_path $data_path \
    --hidden_layers 1 \
    --hidden_features 32 \
    --input_grid_size 8192 \
    --hidden_grid_size 1 \
    --output_grid_size 1 \
    --save_dir logs/Ebach \
    --exp_name fourier1_32_8192_11

# 532 K
python train.py --arch fourier \
    --wav_path $data_path \
    --hidden_layers 1 \
    --hidden_features 64 \
    --input_grid_size 4096 \
    --hidden_grid_size 1 \
    --output_grid_size 1 \
    --save_dir logs/Ebach \
    --exp_name fourier1_64_4096_11

# python train.py --arch fourier \
#     --wav_path data/siren/gt_counting.wav \
#     --save_dir logs/counting \
#     --exp_name fourier_1024

# python train.py --arch fourier \
#     --wav_path data/siren/gt_bach.wav \
#     --save_dir logs/bach \
#     --exp_name fourier_1024

# 198k RELU
# python train.py --arch relu \
#     --wav_path $data_path \
#     --save_dir logs/blues00 \
#     --exp_name relu

# python train.py --arch relu \
#     --wav_path data/siren/gt_counting.wav \
#     --save_dir logs/counting \
#     --exp_name relu

# python train.py --arch relu \
#     --wav_path data/siren/gt_bach.wav \
#     --save_dir logs/bach \
#     --exp_name relu

