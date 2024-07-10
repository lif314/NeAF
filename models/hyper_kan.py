import torch
from torch import nn
import numpy as np


class HyperKANLayer(torch.nn.Module):
    def __init__( self, inputdim, outdim, gridsize=8, addbias=True, smooth_initialization=False):
        super(HyperKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        self.grid_norm_factor = (torch.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        
        # self.fouriercoeffs = torch.nn.Parameter( torch.randn(1, outdim, inputdim, gridsize) / 
        #                                         (np.sqrt(inputdim) * self.grid_norm_factor ) )
        
        print("Init AB---------------")
        self.fouriercoeffs = torch.nn.Parameter(torch.empty(2, outdim, inputdim, gridsize))
        a = np.sqrt(3 / (self.inputdim * self.gridsize ))
        torch.nn.init.uniform_(self.fouriercoeffs, -a, a)

        # #  self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
        #                                     #  np.sqrt(6 / self.in_features) / self.omega_0)

    

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
        # k = 30
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        #This should be fused to avoid materializing memory
        c = torch.cos( k * xrshp )
        s = torch.sin( k * xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum( c * self.fouriercoeffs[0:1],(-2,-1)) 
        # y += torch.sum( s *self.fouriercoeffs[1:2],(-2,-1))
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
        y = torch.reshape(y, outshape)
        return y
    

class HyperKAN(nn.Module):
    def __init__(self,
                 in_features=1,
                 hidden_features=64,
                 hidden_layers=3,
                 out_features=1,
                 input_grid_size=512,
                 hidden_grid_size=5,
                 output_grid_size=3,
                 outermost_linear=False):
        super().__init__()

        self.net = []

        self.net.append(HyperKANLayer(in_features, hidden_features, gridsize=input_grid_size))
        
        for _ in range(hidden_layers):
            self.net.append(HyperKANLayer(hidden_features, hidden_features, gridsize=hidden_grid_size))

        # self.net.append(FourierKANLayer(hidden_features, out_features, gridsize=output_grid_size))
            
        # self.net.append(FourierKANLayer(hidden_features, hidden_features, gridsize=output_grid_size))
        
        # if outermost_linear:
        #     final_linear = nn.Linear(hidden_features, out_features)
        #     self.net.append(final_linear)
        #     print("Output Linear---------------")
        # else:
        self.net.append(HyperKANLayer(hidden_features, out_features, gridsize=output_grid_size))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        t: (B, 1) #  (normalized)
        """

        return self.net(x)