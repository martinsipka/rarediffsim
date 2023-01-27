import torch
import torch.nn as nn
from torch import Tensor

from functorch import grad

class SimpleBias(nn.Module):
    
    
    def __init__(self, n_in=2, neurons = 10, descriptors=None, device="cpu"):
        super(SimpleBias, self).__init__()
               
        activation = nn.SiLU()
        self.descriptors=descriptors

        self.net = nn.Sequential(
            nn.Linear(n_in, neurons),
            activation,
            nn.Linear(neurons, neurons),
            activation,
            #nn.Linear(neurons, neurons),
            #activation,
            nn.Linear(neurons, neurons),
            activation,
            nn.Linear(neurons, 1, bias=False),
        )
        
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
        b = self.net(x)

        return torch.squeeze(b)
        

