import torch
import torch.nn as nn
        
#For linspace
import numpy as np
from functorch import grad

#Bias from dihedrals. It is periodic from -pi to pi.
class BasisBias(nn.Module):

    def __init__(self, H, n_in, Lx, Hx, resolution=10, var=1, neurons=50, descriptors=None, device="cpu", non_linear=True):
        super().__init__()
        self.in_dim = n_in
    
        msh_Lx = Lx + (Hx-Lx)/resolution
        self.means = torch.linspace(msh_Lx, Hx, resolution).reshape(1,1,-1).to(device)
       
        self.vars = (var*torch.ones_like(self.means)).to(device)
        self.descriptors=descriptors
        

        activation = nn.SiLU()

        if non_linear:
            self.net = nn.Sequential(
                nn.Linear(n_in*resolution, neurons),
                activation,
                nn.Linear(neurons, neurons),
                activation,
                #nn.Linear(neurons, neurons),
                #activation,
                nn.Linear(neurons, neurons),
                activation,
                nn.Linear(neurons, 1, bias=False),)
        else:
            self.net = nn.Sequential(nn.Linear(n_in*resolution, 1, bias=False))
        
        self.force_func = lambda R: -grad(self.bias_from_desc)(R)
        
    
    def forward(self, x, training=False):
        b = self.bias_from_desc(x)
        f, = torch.autograd.grad(b.sum(), x, create_graph=training, retain_graph=training)
        return -f
        
    
    def bias_value(self, x, y):
        z = torch.stack((x.reshape(-1,1),y.reshape(-1,1)), axis=1).squeeze()
        return self.bias(z)
    
    def bias_from_desc(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(0)
        desc = self.descriptors.get_descriptors(x)  
        b = self.bias(desc)
        return b
    
    def bias(self, x):
        x = x.unsqueeze(-1)

        d = self.descriptors.metric_from(x, self.means, r_to_desc=False, c_to_desc=False)

        basis = torch.exp(-d**2/(2*self.vars))
        basis = basis.reshape(-1, basis.shape[-1]*basis.shape[-2])
        b = self.net(basis)

        return torch.squeeze(b)
        
