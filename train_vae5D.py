import os

import torch
import torch.nn as nn

import pandas as pd

import matplotlib.pyplot as plt
#Define out simple CV net
from models.simple_cv_vae import SimpleCVVAE

import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument('--save_figs', action='store_true')
parser.add_argument('--no_adjoint', action='store_true')
parser.add_argument("--plot_every", default=1, type=int, help="When to plot figures.")

# Main simulation parameters
# Descriptor loss
parser.add_argument("--iterations", default=20001, type=int, help="Number of iterations to generate training data.")
parser.add_argument("--warmup", default=1200, type=int, help="Warm-up period for the trajectory termalization")
parser.add_argument("--backsteps", default=190, type=int, help="Number of backsteps. Must be smaller as warmup")
parser.add_argument("--plot_nth", default=100, type=int, help="Plotting every nth point.")


parser.add_argument("--batch_size", default=300, type=int, help="Batch size.")
parser.add_argument("--epochs", default=300, type=int, help="Number of training epochs.")
parser.add_argument("--save_steps", default=1, type=int, help="When to save tensor")
parser.add_argument("--dt", default=1.0, type=float, help="Timestep in fs")
parser.add_argument("--barrier", default=1.0, type=float, help="Barier size.")
parser.add_argument("--bias", default="simple", type=str, help="Biased potential")
parser.add_argument("--dimension", default=5, type=int, help="Biased potential")
parser.add_argument("--neurons", default=150, type=int, help="Neurons in a net")
parser.add_argument("--loss", default="quad", type=str, help="Quadratic function")
parser.add_argument("--p_in_domain", default=0.05, type=float, help="Tolerance to it the target. Multiple of variance")

#Simulation controls
parser.add_argument("--temperature", default=300, type=float, help="System temperature.")
parser.add_argument("--gamma", default=1.0, type=float, help="Friction in langevin.")

#Training parameters
parser.add_argument("--learning_factor", default=2.0, type=float, help="Learning rate.")
parser.add_argument("--mini_batch", default=120, type=int, help="Mini batch")
parser.add_argument('--use_non_batched', action='store_true')
parser.add_argument('--device', default="cuda:2", type=str, help="Which device to use. CPU or GPU?")
parser.add_argument('--save_every_model', default=1, type=int, help="When to save the bias potential")

#This training
parser.add_argument("--lr", default=1e-4, type=float, help="VAE learning rate")
parser.add_argument("--vae_batch_size", default=20, type=int, help="VAE batch size")

args = parser.parse_args("")

args.save_figs = True

args.folder = "resultsMB/"+args.bias+ "D" + str(args.dimension)+"p:"+str(args.p_in_domain)+"I:" \
    + str(args.iterations) \
    + "N"+ str(args.neurons) + "dt" + str(args.dt) + "T" + str(args.temperature) + "g" \
    + str(args.gamma) +"/"
isExist = os.path.exists(args.folder)
print(args.folder)
assert isExist
 
interval = 7.5       
n_in = args.dimension

#Generate data


#data = pd.read_csv("converged_results5DMullerBrown.csv").to_numpy()[:,1:]
df = pd.read_csv(args.folder+"converged_results.csv", index_col=0)

data = torch.tensor(df.to_numpy()[:,:n_in], dtype=torch.float32)
print(data)

input_var, input_mean = torch.var_mean(data, axis=0)

dataset = torch.utils.data.TensorDataset((data))

loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = SimpleCVVAE(n_in=n_in, lr=args.lr, input_mean=input_mean, input_var=input_var)


epochs = 15
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

df.to_csv(args.folder + "convergedCV.csv")

model.set_cv_standard(torch.max(cvs), torch.min(cvs))

torch.save(model,args.folder+"cv_model.pt")


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


