import os
import sys

import torch
import torch.nn as nn

import pandas as pd

import matplotlib.pyplot as plt
#Define out simple CV net

sys.path.append('../')

from simple_cv_vae import SimpleCVVAE
 
interval = 7.5    
   
n_in = 5
epochs = 40
batch_size = 50
lr = 1e-4

#Generate data


#data = pd.read_csv("converged_results5DMullerBrown.csv").to_numpy()[:,1:]
df = pd.read_csv("converged_results.csv", index_col=0)

data = torch.tensor(df.to_numpy()[:,:n_in], dtype=torch.float32)


input_var, input_mean = torch.var_mean(data, axis=0)

dataset = torch.utils.data.TensorDataset((data))

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SimpleCVVAE(n_in=n_in, lr=lr, input_mean=input_mean, input_var=input_var)



loss_count = 0
for e in range(epochs):
    for batch_ndx, sample in enumerate(loader):
        sample, = sample
        latent_loss, r_loss, tot_loss = model.train_batch(sample)
        loss_count += tot_loss.detach().numpy()
    
    print(e, loss_count/len(loader))
    loss_count=0


cvs = model.cv(data)
df["cvs"] = cvs.detach().numpy()

df.to_csv("convergedCV.csv")

model.set_cv_standard(torch.max(cvs), torch.min(cvs))

torch.save(model,"cv_model.pt")


if data.shape[1] == 2:
    resolution = 30
    x, y = torch.meshgrid(torch.linspace(0, 50, resolution), torch.linspace(0, 50, resolution))
    x.requires_grad = True
    y.requires_grad = True
    data = torch.cat((x.reshape(-1,1), y.reshape(-1,1)), axis=1)

    cv = model.cv(data)

    u, = torch.autograd.grad(torch.sum(cv), x, create_graph=True)
    v, = torch.autograd.grad(torch.sum(cv), y)

    u = u.reshape(resolution,resolution).detach()
    v = v.reshape(resolution,resolution).detach()


    fig, ax = plt.subplots()

    c = ax.pcolormesh(x.reshape(resolution,resolution).detach(), y.reshape(resolution,resolution).detach(), cv.reshape(resolution,resolution).detach(), cmap='RdBu', vmin=torch.min(cv), vmax=torch.max(cv.detach()))
    ax.quiver(x.detach(),y.detach(),u,v, color='blue')
    ax.set_title('pcolormesh')
    ax.axis([x.detach().min(), x.detach().max(), y.detach().min(), y.detach().max()])
    # set the limits of the plot to the limits of the data

    fig.colorbar(c, ax=ax)

    plt.show()


