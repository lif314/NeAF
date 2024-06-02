import torch
import torchaudio.transforms as T
import numpy as np

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

@torch.no_grad()
def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


# https://en.wikipedia.org/wiki/Signal-to-noise_ratio
# https://github.com/kuleshov/audio-super-res/blob/master/src/models/model.py
@torch.no_grad()
def calc_snr(pred, gt):
    P_signal = torch.mean(gt ** 2)
    P_noise = torch.mean((gt - pred) ** 2)
    snr = 20 * torch.log(torch.sqrt(P_signal) / torch.sqrt(P_noise + 1e-6) + 1e-8) / np.log(10.)
    return snr

def get_power(x):
    S = T.Spectrogram(n_fft=2048)(x)
    S = torch.log(torch.abs(S)**2 + 1e-8) / np.log(10)
    return S

def compute_log_distortion(x_pr, x_hr):
    x_hr = torch.flatten(x_hr)
    x_pr = torch.flatten(x_pr)
    S1 = get_power(x_hr)
    S2 = get_power(x_pr)
    lsd = torch.mean(torch.sqrt(torch.mean((S1-S2)**2 + 1e-8, dim=1)), dim=0)
    return min(lsd, 10.)