import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io.wavfile as wavfile

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class AudioDataset(Dataset):
    def __init__(self, wav_path: str = "data/audio/gt_bach.wav"):
        rate, data = wavfile.read(wav_path)
        
        amplitude = data.astype(np.float32)
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        
        # timepoints
        self.rate = rate
        self.timepoints = get_mgrid(len(data), 1) # [N, 1]
        self.amplitude = torch.Tensor(amplitude).view(-1, 1) # [N, 1]

    def __len__(self):
        return self.timepoints.shape[0]

    def __getitem__(self, idx: int):
        return {"t": self.timepoints[idx], "a": self.amplitude[idx]}