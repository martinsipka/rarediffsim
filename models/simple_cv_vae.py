import torch
import torch.nn as nn
import schnetpack as spk

class SimpleCVVAE(nn.Module):
    
    
    def __init__(self, n_in=2, lr=1e-4, input_mean=torch.tensor(0.), input_var=torch.tensor(1.)):
        super(SimpleCVVAE, self).__init__()
        
        activation = nn.Softplus()
        self.net = nn.Sequential(
            spk.nn.base.Standardize(input_mean, torch.sqrt(input_var)),
            nn.Linear(n_in, 50),
            activation,
            nn.Linear(50, 50),
            activation,
            nn.Linear(50, 50),
            activation,
            nn.Linear(50, 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1, 50),
            activation,
            nn.Linear(50, 50),
            activation,
            nn.Linear(50, 50),
            activation,
            nn.Linear(50, n_in),
        )
        
        self.n_in = n_in
        
        self.recon_loss = nn.MSELoss()
        self.standardize_cv = spk.nn.base.Standardize(torch.tensor(0.), torch.tensor(1.))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def set_cv_standard(self, cv_maxima, cv_minima):
        scale = (cv_maxima - cv_minima) * 0.5
        shift = (cv_maxima + cv_minima) / 2
        self.standardize_cv = spk.nn.base.Standardize(shift, scale)

    def simple_kl_divergence(a_mean, a_sd):
        """Method for computing KL divergence of two normal distributions."""
        #N, d = A.shape

        #define zero loss at start

        a_sd_squared = a_sd ** 2

        b_mean = 0#     = torch.zeros_like(a_mean)

        b_sd_squared = 1# = torch.ones_like(a_sd) ** 2

        ratio = a_sd_squared / b_sd_squared + 1E-9

        loss = torch.mean((a_mean - b_mean) ** 2 / (2 * b_sd_squared)
                + (ratio - torch.log(ratio) - 1) / 2)
        return loss

    
    def encode(self,x):
        out = self.net(x)
        return out[:,0], out[:,1]
        
    def cv(self,x):
        x = x.reshape((-1, self.n_in))
        x = self.net(x)[:,0]
        x = self.standardize_cv(x)
        return x
        
    def decode(self,x):
        return self.decoder(x)
        
    def loss_ascend(self, x):
        s, _ = self.encode(x.reshape(-1,self.n_in))
        y = self.decode(s.reshape(-1,1))
        loss = self.recon_loss(y, x.reshape(-1,self.n_in))
        g, = torch.autograd.grad(loss, x)
        return torch.mean((y-x)**2, axis=1), g
        
    def biasf(self, x):
        s, _ = self.encode(x)
        g, = torch.autograd.grad(s, x)
        return g
        
    def train_batch(self, x):
        self.optimizer.zero_grad()
        means, logvars = self.encode(x)

        noise = torch.clone(means).normal_(0, 1)
        z = noise * torch.exp(logvars / 2) + means
        # =================latent loss=======================
        latent_loss = SimpleCVVAE.simple_kl_divergence(means, torch.exp(logvars / 2))
        #print(z)
        # =================decode=====================

        decoded = self.decode(z.reshape(-1,1))
        

        r_loss = self.recon_loss(decoded, x)
        tot_loss = latent_loss + self.n_in*r_loss
        tot_loss.backward()
        self.optimizer.step()
        return latent_loss, r_loss, tot_loss
