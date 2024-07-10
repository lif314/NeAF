import torch
from torch import nn
import numpy as np


class FourierKANLayer(torch.nn.Module):
    def __init__( self, inputdim, outdim, gridsize=8, addbias=True, smooth_initialization=False, 
                 is_first=False, init_type="uniform"):
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
       
        print("Init type===========: ", init_type)
        # Norm
        if init_type == "norm":
            # if is_first:
            std_dev = np.sqrt(1.0 / (self.inputdim * self.gridsize))
            # else:
            # sigma = self.inputdim * self.gridsize + self.outdim * np.sum(np.arange(1, self.gridsize + 1)**2)
            # std_dev = np.sqrt(2.0 / sigma)
            # sigma = self.outdim * np.sum(np.arange(1, self.gridsize + 1)**2)
            # std_dev = np.sqrt(1.0 / sigma)
            self.fouriercoeffs = torch.nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) * std_dev)
        
        # Uniform
        elif init_type == "uniform":
            # if is_first:
                # uniform_range = np.sqrt(3.0 / (self.inputdim * self.gridsize))
            # else:
            # sigma = self.inputdim * self.gridsize + self.outdim * np.sum(np.arange(1, self.gridsize + 1)**2)
            # uniform_range = np.sqrt(6.0 / sigma)
            sigma = self.outdim * np.sum(np.arange(1, self.gridsize + 1)**2)
            uniform_range = np.sqrt(3.0 / sigma)
            self.fouriercoeffs = torch.nn.Parameter(torch.FloatTensor(2, outdim, inputdim, gridsize).uniform_(-uniform_range, uniform_range))

        # Random
        elif init_type == "rand":
            self.fouriercoeffs = torch.nn.Parameter(torch.rand(2, outdim, inputdim, gridsize))

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
        y =  torch.sum( c * self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum( s * self.fouriercoeffs[1:2],(-2,-1))
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
                 output_grid_size=3,
                 outermost_linear=False,
                 init_type="uniform"):
        super().__init__()

        self.net = []
       
        self.net.append(FourierKANLayer(in_features, hidden_features, gridsize=input_grid_size, is_first=True, init_type=init_type))
        
        for _ in range(hidden_layers):
            self.net.append(FourierKANLayer(hidden_features, hidden_features, gridsize=hidden_grid_size, init_type=init_type))

        self.net.append(FourierKANLayer(hidden_features, out_features, gridsize=output_grid_size, init_type=init_type))
        
        # if outermost_linear:
        #     final_linear = nn.Linear(hidden_features, out_features)
            
        #     self.net.append(final_linear)
        # else:
        #     self.net.append(FourierKANLayer(hidden_features, out_features, gridsize=output_grid_size))
        

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        t: (B, 1) #  (normalized)
        """
        return self.net(x)