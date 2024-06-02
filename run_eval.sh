# python train.py --arch siren --exp_name siren

# 264K
# python train.py --arch kan --hidden_layers 4 --hidden_features 64 --exp_name fourier_4_64
# 198K
# python train.py --arch kan --hidden_layers 3 --hidden_features 64 --exp_name fourier_3_64
# 133K
# python train.py --arch kan --hidden_layers 2 --hidden_features 64 --exp_name fourier_2_64
# 266K
# python train.py --arch kan --hidden_layers 1 --hidden_features 128 --exp_name fourier_1_128


# baselines
python train.py --arch relu --exp_name relu
python train.py --arch gaussian --exp_name gaussian
python train.py --arch quadratic --exp_name quadratic
python train.py --arch laplacian --exp_name laplacian
