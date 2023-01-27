import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


def plot_intermediate(traj_tensor, traj_desc, known_cvs, sample_bias_force, i, args, domain, potential=None):
    
    Lx,Ly = domain["Lx"], domain["Ly"]
    Hx,Hy = domain["Hx"], domain["Hy"]
    resolution = domain["res"]
    
    sample = traj_tensor[:,::args.plot_nth].reshape(-1, *traj_tensor.shape[2:])
    desc_sample = traj_desc[:,::args.plot_nth].reshape(-1, *traj_desc.shape[2:])
    react_cv_sample = known_cvs[:args.batch_size,::args.plot_nth].reshape(-1,2)
    prod_cv_sample = known_cvs[args.batch_size:,::args.plot_nth].reshape(-1,2)
    cv_sample = known_cvs[:,::args.plot_nth].reshape(-1,2)
    
    x, y = torch.meshgrid(torch.linspace(Lx, Hx, resolution), torch.linspace(Ly, Hy, resolution), indexing='ij')
    x_dev, y_dev = x.to(args.device), y.to(args.device)
    
    if args.dimension == 2:
        
        z_b = sample_bias_force.bias_value(x_dev,y_dev).reshape((resolution,resolution)).detach().cpu()
        B_min, B_max = z_b.min(), z_b.max()

    else:
        

        bias = sample_bias_force.bias_from_desc(sample.requires_grad_(True).to(args.device)).detach().cpu()
        points = torch.stack((cv_sample[:,0], cv_sample[:,1]), axis=-1).detach().cpu()
        z_b = griddata(points, bias, (x, y), method='linear')
        B_min, B_max = bias.min(), bias.max()
        
   
    plt.figure(figsize = (16,6)) 
    plt.subplot(121)
    c_bias = plt.pcolormesh(x, y, z_b, cmap='magma', vmin=B_min, vmax=B_max, shading='auto')
    plt.colorbar(c_bias)
    plt.axis([Lx, Hx, Ly, Hy])
    plt.title('B(x)')
    
    
    plt.subplot(122)
    
    alpha = 1.0
    if potential:
        z = potential.U_split(x_dev, y_dev).reshape((resolution,resolution)).detach().cpu()
        c_bias = plt.pcolormesh(x, y, z, cmap='magma', vmin=-10, vmax=10, shading='auto')
        alpha = 0.5
    
    plt.hexbin(cv_sample[:,0], cv_sample[:,1].detach().cpu(), bins="log", alpha=alpha, cmap="viridis",gridsize=40, mincnt=5)
    plt.colorbar()
    plt.axis([Lx, Hx, Ly, Hy])
                
    if args.save_figs:
        fig = plt.gcf()
        fig.savefig(args.folder+"result"+str(i))
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

