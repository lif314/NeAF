import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav_path', type=str, default='data/audio/gt_bach.wav',
                        help='path to the image to reconstruct')
    parser.add_argument('--use_pe', default=False, action='store_true',
                        help='use positional encoding or not')
    parser.add_argument('--arch', type=str, default='identity',
                        choices=['fourier', 'relu', 'ff', 'siren',
                                 'gaussian', 'quadratic', 'multi-quadratic',
                                 'laplacian', 'super-gaussian', 'expsin'],
                        help='network structure')
    
    parser.add_argument('--in_features', type=int, default=1,
                        help='input dim of Network')
    parser.add_argument('--out_features', type=int, default=1,
                        help='output dim of Network')
    parser.add_argument('--hidden_layers', type=int, default=4,
                        help='number of KAN layers')
    parser.add_argument('--hidden_features', type=int, default=64,
                        help='number of KAN hidden layer dim')
    parser.add_argument('--input_grid_size', type=int, default=512,
                        help='number of KAN grid zise of first layer')
    parser.add_argument('--hidden_grid_size', type=int, default=5,
                        help='number of KAN grid zise of hidden layers')
    parser.add_argument('--output_grid_size', type=int, default=3,
                        help='number of KAN grid zise of output layer')
    parser.add_argument('--a', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=1.)
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')

    parser.add_argument('--sc', type=float, default=0.1, # default 10.
                        help='fourier feature scale factor (std of the gaussian)')
    parser.add_argument('--first_omega_0', type=float, default=3000.,
                        help='omega in siren')
    parser.add_argument('--hidden_omega_0', type=float, default=30.,
                        help='omega in siren')

    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')

    parser.add_argument('--save_dir', type=str, default='logs',
                        help='experiment log dir')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()