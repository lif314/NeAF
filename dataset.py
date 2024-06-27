import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class AudioDataset(Dataset):
    def __init__(self,  dataset_name = "gtzan", audio_path: str = "data/audio/gt_bach.wav"):
        # print("dataset: ", dataset_name)
        print("dataset seq: ",audio_path)
        
        # if dataset_name == 'gtzan':
            # rate, data = wavfile.read(audio_path)
        # elif dataset_name == 'libri':
        data, rate = sf.read(audio_path, dtype='float32')
        print("rate: ", rate)
        print("samples: ", len(data))

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
    
