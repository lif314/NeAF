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
    snr = 20 * torch.log10(torch.sqrt(P_signal) / torch.sqrt(P_noise + 1e-8) + 1e-8)
    return snr

def get_power(x):
    # S = T.Spectrogram(n_fft=2048)(x)
    # S = torch.log(torch.abs(S)**2 + 1e-8) / np.log(10)
    S = torch.stft(x, n_fft=2048, return_complex=True)
    S = torch.log10(torch.abs(S)**2 + 1e-8)
    return S

def compute_log_distortion(x_pr, x_hr):
    x_hr = torch.flatten(x_hr)
    x_pr = torch.flatten(x_pr)
    S1 = get_power(x_hr)
    S2 = get_power(x_pr)
    lsd = torch.mean(torch.sqrt(torch.mean((S1-S2)**2 + 1e-8, dim=1)), dim=0)
    return min(lsd, 10.)


from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics import PermutationInvariantTraining
from torchmetrics.functional import scale_invariant_signal_distortion_ratio
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.sdr import SignalDistortionRatio
from torchmetrics import SignalNoiseRatio


def audio_metrics(preds, target, rate, device='cpu'):

    preds = preds.to(device)
    target = target.to(device)

    print("pred shape: ", preds.shape)
    print("gt shape: ", target.shape)
    print("rate: ", rate)

    metrics = {}

    min_length = rate // 4  # minimum length for PESQ, 1/4 second

    if preds.shape[-1] < min_length or target.shape[-1] < min_length:
        print("Input signals are too short for PESQ calculation.")
        metrics['nb_pesq'] = float('nan')
        metrics['wb_pesq'] = float('nan')
    else:
        # narrow-band PESQ 8000Hz
        nb_resampler = T.Resample(orig_freq=rate, new_freq=8000).to(device)
        nb_pesq = PerceptualEvaluationSpeechQuality(8000, 'nb').to(device)
        nb_pesq_score = nb_pesq(nb_resampler(preds), nb_resampler(target))
        metrics['nb_pesq'] = nb_pesq_score

        # wide-band PESQ 16000 Hz
        wb_resampler = T.Resample(orig_freq=rate, new_freq=16000).to(device)
        wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb').to(device)
        wb_pesq_score = wb_pesq(wb_resampler(preds), wb_resampler(target))
        metrics['wb_pesq'] = wb_pesq_score

    # PIT
    # pit = PermutationInvariantTraining(ScaleInvariantSignalDistortionRatio(), 'permutation-wise').to(device)
    # pit_score = pit(preds, target)
    # metrics['pit'] = pit_score

    # SI-SDR
    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    si_sdr_score = si_sdr(preds, target)
    metrics['si_sdr'] = si_sdr_score

    # SI-SNR
    si_snr = ScaleInvariantSignalNoiseRatio().to(device)
    si_snr_score = si_snr(preds, target)
    metrics['si_snr'] = si_snr_score

    # STOI
    # nb_resampler = T.Resample(orig_freq=rate, new_freq=8000).to(device)
    # stoi_nb_fn = ShortTimeObjectiveIntelligibility(8000, False).to(device)
    # stoi_nb_score = stoi_nb_fn(nb_resampler(preds), nb_resampler(target))
    # metrics['stoi_nb'] = stoi_nb_score

    # SDR
    # sdr = SignalDistortionRatio().to(device)
    # sdr_score = sdr(preds, target)
    # metrics['sdr'] = sdr_score

    # SNR
    snr = SignalNoiseRatio().to(device)
    snr_score = snr(preds, target)
    metrics['snr'] = snr_score

    return metrics



