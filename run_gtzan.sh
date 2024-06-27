data_path=data/gtzan/genres/blues/blues.00000.wav


python train.py --arch siren \
    --save_dir logs_blues001 \
    --exp_name siren \
    --dataset_name gtzan \
    --audio_path $data_path


python train.py --arch fourier \
    --save_dir logs_blues001 \
    --exp_name fourier \
    --dataset_name gtzan \
    --audio_path $data_path

# 336 K
python train.py --arch fourier \
    --input_grid_size 2048 \
    --hidden_grid_size 3 \
    --output_grid_size 1 \
    --save_dir logs_blues001 \
    --exp_name fourier_2048 \
    --dataset_name gtzan \
    --audio_path $data_path
