import numpy as np
import torch
from torch import nn

class IncodeSineLayer(nn.Module):
    '''
    SineLayer is a custom PyTorch module that applies a modified Sinusoidal activation function to the output of a linear transformation
    with adjustable parameters.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If True, initializes the weights with a narrower range. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.
        
    Additional Parameters:
        a_param (float): Exponential scaling factor for the sine function. Controls the amplitude. 
        b_param (float): Exponential scaling factor for the frequency.
        c_param (float): Phase shift parameter for the sine function.
        d_param (float): Bias term added to the output.

    '''
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, a_param, b_param, c_param, d_param):
        output = self.linear(input)
        output = torch.exp(a_param) * torch.sin(torch.exp(b_param) * self.omega_0 * output + c_param) + d_param
        return output
    

class INCODE(nn.Module):
    def __init__(self, 
                 in_features=1,
                 hidden_features=4,
                 hidden_layers=256,
                 out_features=1,
                 outermost_linear=True,
                 first_omega_0=3000,
                 hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(IncodeSineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(IncodeSineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(IncodeSineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords):
        return self.net(coords)