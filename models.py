import torch
from torch import nn
import numpy as np
import torchaudio
import math

class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out=3,
                 n_layers=4,
                 n_hidden_units=256,
                 act='relu',
                 act_trainable=False,
                 **kwargs):
        super().__init__()

        layers = []
        for i in range(n_layers):

            if i == 0:
                l = nn.Linear(n_in, n_hidden_units)
            elif 0 < i < n_layers-1:
                l = nn.Linear(n_hidden_units, n_hidden_units)

            if act == 'relu':
                act_ = nn.ReLU(True)
            elif act == 'gaussian':
                act_ = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'quadratic':
                act_ = QuadraticActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'multi-quadratic':
                act_ = MultiQuadraticActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'laplacian':
                act_ = LaplacianActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'super-gaussian':
                act_ = SuperGaussianActivation(a=kwargs['a'], b=kwargs['b'],
                                               trainable=act_trainable)
            elif act == 'expsin':
                act_ = ExpSinActivation(a=kwargs['a'], trainable=act_trainable)

            if i < n_layers-1:
                layers += [l, act_]
            else:
                layers += [nn.Linear(n_hidden_units, n_out), nn.Sigmoid()]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 2) # pixel uv (normalized)
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org

        return {'model_in': coords_org, 'model_out': self.net(coords)}  # (B, 3) rgb


#This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
#It should be easier to optimize as fourier are more dense than spline (global vs local)
#Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
#The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
#Avoiding the issues of going out of grid

class FourierKANLayer(torch.nn.Module):
    def __init__( self, inputdim, outdim, gridsize=8, addbias=True, smooth_initialization=False):
        super(FourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        self.grid_norm_factor = (torch.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
       
        self.fouriercoeffs = torch.nn.Parameter( torch.randn(2, outdim, inputdim, gridsize) / 
                                                (np.sqrt(inputdim) * self.grid_norm_factor ) )
    
        # self.k = nn.Parameter(torch.randn(1, 1, 1, 1) * self.omega)

        if( self.addbias ):
            self.bias  = torch.nn.Parameter( torch.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = torch.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        #This should be fused to avoid materializing memory
        c = torch.cos( k * xrshp )
        s = torch.sin( k * xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum( c *self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum( s *self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape( y, outshape)
        return y
    

class FourierKAN(nn.Module):
    def __init__(self,
                 in_features=1,
                 hidden_features=64,
                 hidden_layers=3,
                 out_features=1,
                 input_grid_size=512,
                 hidden_grid_size=5,
                 output_grid_size=3
                 ):
        super().__init__()

        self.net = []
       
        self.net.append(FourierKANLayer(in_features, hidden_features, gridsize=input_grid_size))
        
        for _ in range(hidden_layers):
             self.net.append(FourierKANLayer(hidden_features, hidden_features, gridsize=hidden_grid_size))

        self.net.append(FourierKANLayer(hidden_features, out_features, gridsize=output_grid_size))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        t: (B, 1) #  (normalized)
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        
        output = None
        if x.dim() == 3:
            coords = coords.squeeze(0)
            output = self.net(coords).unsqueeze(0)
        else:
            output = self.net(coords)
        
        return {'model_in': coords_org, 'model_out':  output}

class PE(nn.Module):
    """
    perform positional encoding
    """
    def __init__(self, P):
        """
        P: (2, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1]*2

    def forward(self, x):
        """
        x: (B, 2)
        """
        x_ = 2*np.pi*x @ self.P # (B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], 1) # (B, 2*F)
    

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None, scale=2):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.sidelength = sidelength
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies
        # self.frequencies_per_axis = (num_frequencies * np.array(sidelength)) // max(sidelength)
        self.out_dim = in_features + in_features * 2 * self.num_frequencies  # (sum(self.frequencies_per_axis))

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(np.floor(np.log2(nyquist_rate)))

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):

            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((self.scale ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((self.scale ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
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
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self,
                 in_features=1,
                 hidden_features=256,
                 hidden_layers=3,
                 out_features=1,
                 outermost_linear=True, 
                 first_omega_0=3000,
                 hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return {'model_in': coords, 'model_out': output}   


# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))