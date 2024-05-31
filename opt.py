import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav_path', type=str, default='data/audio/gt_bach.wav',
                        help='path to the image to reconstruct')
    parser.add_argument('--use_pe', default=False, action='store_true',
                        help='use positional encoding or not')
    parser.add_argument('--arch', type=str, default='identity',
                        choices=['kan', 'relu', 'ff', 'siren', 'gabor', 'bacon',
                                 'gaussian', 'quadratic', 'multi-quadratic',
                                 'laplacian', 'super-gaussian', 'expsin'],
                        help='network structure')
    parser.add_argument('--use_kan_pe', default=False, action='store_true',
                        help='use KAN for PE or not')
    parser.add_argument('--kan_layers', type=int, default=4,
                        help='number of KAN layers')
    parser.add_argument('--n_out', type=int, default=1,
                        help='out dim of Network')
    parser.add_argument('--kan_hidden_dim', type=int, default=64,
                        help='number of KAN hidden layer dim')
    parser.add_argument('--kan_basis', type=str, default='bspline',
                        choices=['bspline','grbf', 'rbf', 'fourier', 'fcn', 'fcn_interpo',
                                 'chebyshev', 'jacobi', 'bessel',
                                 'chebyshev2', 'finonacci', 'hermite',
                                 'legendre', 'gegenbauer', 'lucas',
                                 'laguerre', 'mexican_hat', 'morlet',
                                 'dog', 'meyer', 'shannon', 'bump'],
                        help='KAN basis functions')
    parser.add_argument('--a', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=1.)
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')

    parser.add_argument('--sc', type=float, default=0.1, # default 10.
                        help='fourier feature scale factor (std of the gaussian)')
    parser.add_argument('--omega_0', type=float, default=30.,
                        help='omega in siren')

    parser.add_argument('--batch_size', type=int, default=8092,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')

    parser.add_argument('--save_dir', type=str, default='logs/audio',
                        help='experiment log dir')
    parser.add_argument('--exp_name', type=str, default='siren',
                        help='experiment name')

    return parser.parse_args()