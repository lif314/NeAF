import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out=1,
                 n_layers=6,
                 n_hidden_units=256,
                 act='relu',
                 act_trainable=False,
                 **kwargs):
        super().__init__()

        layers = []
        for i in range(n_layers):

            if i == 0:
                l = nn.Linear(n_in, n_hidden_units, bias=True)
            elif 0 < i < n_layers-1:
                l = nn.Linear(n_hidden_units, n_hidden_units, bias=True)

            if act == 'relu':
                act_ = nn.ReLU(inplace=True)
            if act == 'tanh':
                act_ = nn.Tanh()
            if act == 'softsign':
                act_ = nn.Softsign()
            if act == 'sinc':
                act_ = SincActivation(a=kwargs['a'], trainable=act_trainable)
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
                # layers += [nn.Linear(n_hidden_units, n_out), act_]
                layers += [nn.Linear(n_hidden_units, n_out, bias=True)]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 2) # pixel uv (normalized)
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org

        return {'model_in': coords_org, 'model_out': self.net(coords)}  # (B, 3) rgb



# different activation functions
class SincActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        # self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        # Avoid division by zero
        x = torch.where(x == 0, torch.Tensor(1e-7), x)
        return torch.sin(torch.pi * x) / (torch.pi * x)

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