python train.py --arch siren  --exp_name siren --save_dir logs_counting --wav_path data/audio/gt_counting.wav


python train.py --arch kan --hidden_layers 5 --hidden_features 64 --exp_name fourier_5_64 --save_dir logs_counting --wav_path data/audio/gt_counting.wav
python train.py --arch kan --hidden_layers 3 --hidden_features 64 --exp_name fourier_3_64 --save_dir logs_counting --wav_path data/audio/gt_counting.wav
python train.py --arch kan --hidden_layers 4 --hidden_features 64 --exp_name fourier_4_64 --save_dir logs_counting --wav_path data/audio/gt_counting.wav


python train.py --arch kan --hidden_layers 5 --hidden_features 64 --exp_name fourier_5_64_3K --save_dir logs_counting --wav_path data/audio/gt_counting.wav --num_epochs 3000
python train.py --arch kan --hidden_layers 4 --hidden_features 64 --exp_name fourier_4_64_3K --save_dir logs_counting --wav_path data/audio/gt_counting.wav --num_epochs 3000
python train.py --arch siren  --exp_name siren_3K --save_dir logs_counting --wav_path data/audio/gt_counting.wav --num_epochs 3000

# 264K
# 198K
# 133K
# 266K
# python train.py --arch kan --hidden_layers 1 --hidden_features 128 --exp_name fourier_1_128


# baselines
# python train.py --arch relu --exp_name relu
# python train.py --arch gaussian --exp_name gaussian
# python train.py --arch quadratic --exp_name quadratic
# python train.py --arch laplacian --exp_name laplacian
