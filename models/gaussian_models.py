import torch
import torch.nn as nn
        
#For linspace
import numpy as np
from functorch import grad
        

class GridGaussians(nn.Module):

    def __init__(self, H, n_in, Lx, Hx, resolution, var=1, descriptors=None, device="cpu"):
        super().__init__()
        
        """
        Parameters
        ----------
        H : float or Tensor
            Initial Gaussian height.
        n_in : int
            dimension. Here only 1,2 or 3 make sense. 
        ranges : Tensor (n_in, 2)
            How the CV is bounded. 
        var : float or Tensor optional
            Gaussian width. Default 1.0.
        descriptors : callable optional default: None
            Function providing descriptors to use, transforming coordinates to more invariant and relevant form.
        """
        
        #Currently for Gaussian grid only dim 2 is supported. 
        assert n_in == 2
        
        self.in_dim = n_in
    
        
        msh_Lx = Lx + (Hx-Lx)/resolution

        x,y = torch.meshgrid(torch.linspace(msh_Lx, Hx, resolution), torch.linspace(msh_Lx, Hx, resolution), indexing='ij')
        x, y = x.to(device), y.to(device)
        self.means = torch.stack((x.reshape(-1,1),y.reshape(-1,1)), axis=1).reshape(1,-1,n_in).to(device)

        self.height = torch.nn.Parameter(H*torch.ones(1,self.means.shape[1]).to(device))

        self.vars = (var*torch.ones_like(self.means)).to(device)
        self.descriptors=descriptors
        
        self.force_func = lambda R: -grad(self.bias_from_desc)(R)
        
    
    def forward(self, x, training=False):
        b = self.bias_from_desc(x)
        f, = torch.autograd.grad(b.sum(), x, create_graph=training, retain_graph=training)
        return -f
        
    def gauss(self, x, size, mean, std):

        x=x.reshape((-1,1,self.in_dim))
        return size*torch.exp(torch.sum(-(x-mean)**2/(2*std**2), dim=2))
    
    def bias_value(self, x, y):
        z = torch.stack((x.reshape(-1,1),y.reshape(-1,1)), axis=1).squeeze()
        return self.bias(z)
    
    def bias_from_desc(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(0)
        desc = self.descriptors.get_descriptors(x)    
        b = torch.sum(self.bias(desc))
        return b
    
    def bias(self, x):
        x = x.unsqueeze(-2)
        d_uncor = x - self.means

        d = self.descriptors.metric_from(x, self.means, r_to_desc=False, c_to_desc=False)

        g = self.height*torch.exp(-torch.sum(d**2/(2*self.vars), dim=-1))

        return torch.sum(g, dim=1)
        
