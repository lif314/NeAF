import torch
from torch import nn
import numpy as np
import math


class FourierKANLayer(torch.nn.Module):
    def __init__( self, inputdim, outdim, gridsize=8, addbias=True, smooth_initialization=False):
        super(FourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        print("Inputdim: ", self.inputdim) # 3
        print("Outputdim: ", self.outdim) # 2
        print("Grid size: ", self.gridsize) # 8
        
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
        
        print("fouriercoeffs shape: ", self.fouriercoeffs.shape)  # [2, 2, 3, 8]

        if( self.addbias ):
            self.bias  = torch.nn.Parameter( torch.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        print("input x shape: ", x.shape) # [5, 3]

        outshape = xshp[0:-1]+(self.outdim,)
        x = torch.reshape(x,(-1,self.inputdim))
        
        print("input x shape: ", x.shape) # [5, 3]

        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))

        print("frequency k shape: ", k.shape) # [1, 1, 1, 8]

        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        
        print("input xrshp shape: ", xrshp.shape) # [5, 1, 3, 1]

        #This should be fused to avoid materializing memory
        c = torch.cos( k * xrshp ) # [3, 1] * [1, 8] = [3, 8]
        s = torch.sin( k * xrshp ) # [3, 1] * [1, 8] = [3, 8]
        print("cos shape: ", c.shape) # [5, 1, 3, 8]
        print("sin shape: ", s.shape) # [5, 1, 3, 8]
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        
        phi_c = c *self.fouriercoeffs[0:1]
        print("phi_c shape: ", phi_c.shape)
        y =  torch.sum(phi_c, (-2,-1))

      
        print("b shape: ", self.fouriercoeffs[0:1].shape) # [1, 2, 3, 8]

        y += torch.sum( s *self.fouriercoeffs[1:2],(-2,-1))
        print("a shape: ", self.fouriercoeffs[1:2].shape) # [1, 2, 3, 8]

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

        print("middle y shape: ", y.shape) # [5, 2]
        y = torch.reshape( y, outshape)
        
        print("output y shape: ", y.shape) # [5, 2]
        return y
    

# Define input and output dimensions
inputdim = 3
outdim = 2

# Create an instance of FourierKANLayer
layer = FourierKANLayer(inputdim=inputdim, outdim=outdim, gridsize=8, addbias=True, smooth_initialization=False)

# Generate random input data
x = torch.randn(5, inputdim)

# Pass the random input data through the FourierKANLayer
output = layer(x)

# Print the output
print("Output: ", output)