#Get initial descriptors and estimate variance in them. 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import torch

def get_covariance(react_pos, prod_pos, react_target, prod_target, descriptors, domain, args, potential=None):
    reactant_dihedrals = descriptors.metric_from(react_pos.reshape((-1,*react_pos.shape[2:])), react_target)
    product_dihedrals = descriptors.metric_from(prod_pos.reshape((-1,*prod_pos.shape[2:])), prod_target)

    r_d = descriptors.get_descriptors(react_pos.reshape((-1,*react_pos.shape[2:])))
    p_d = descriptors.get_descriptors(prod_pos.reshape((-1,*prod_pos.shape[2:])))

    #Get the normal distribution covariance matrix
    react_cov = torch.cov(reactant_dihedrals.T)
    prod_cov = torch.cov(product_dihedrals.T)

    #Invert it. We are only inverting once in the runtime. 
    react_var_inv = torch.inverse(react_cov).unsqueeze(0)
    prod_var_inv = torch.inverse(prod_cov).unsqueeze(0)

    Lx,Ly = domain["Lx"], domain["Ly"]
    Hx,Hy = domain["Hx"], domain["Hy"]
    resolution = domain["res"]
    x, y = torch.meshgrid(torch.linspace(Lx, Hx, resolution), torch.linspace(Ly, Hy, resolution), indexing='ij')
    
    
    if args.dimension == 2:
        d_r = (torch.stack((x,y),dim=-1)-react_target).reshape(-1,2)
        d_p = (torch.stack((x,y),dim=-1)-prod_target).reshape(-1,2)
        
        summ_r = (d_r.unsqueeze(1) @ react_var_inv @ d_r.unsqueeze(2)).squeeze()
        prod_r = torch.exp(-1/2*summ_r).reshape(resolution, resolution)

        summ_p = (d_p.unsqueeze(1) @ prod_var_inv @ d_p.unsqueeze(2)).squeeze()
        prod_p = torch.exp(-1/2*summ_p).reshape(resolution, resolution)
        
        print(prod_r.shape)
    
    
    else:
        summ_r = (reactant_dihedrals.unsqueeze(1) @ react_var_inv @ reactant_dihedrals.unsqueeze(2)).squeeze()
        prod_r = torch.exp(-1/2*summ_r)
        
        summ_p = (product_dihedrals.unsqueeze(1) @ prod_var_inv @ product_dihedrals.unsqueeze(2)).squeeze()
        prod_p = torch.exp(-1/2*summ_p)
        
        points_r = r_d[:,descriptors.cv_index].detach().cpu()
        points_p = p_d[:,descriptors.cv_index].detach().cpu()
        prod_r = griddata(points_r, prod_r, (x, y), method='linear',fill_value=0.0)
        prod_p = griddata(points_p, prod_p, (x, y), method='linear',fill_value=0.0)

    alpha = 1.0
    if potential:
        pot_image = potential.U_split(x.to(args.device),y.to(args.device)).detach().cpu()
        c_bias = plt.pcolormesh(x, y, pot_image, vmin=-10, vmax=10, cmap='magma', shading='auto')
        alpha = 0.5
    
    c_bias = plt.pcolormesh(x, y, prod_r+prod_p, alpha=alpha, cmap='magma', shading='auto')


    plt.title('Potential and hitboxes')
    plt.axis([Lx, Hx, Ly, Hy])
    plt.colorbar(c_bias)
    
    if args.save_figs:
        fig = plt.gcf()
        fig.savefig(args.folder+"hitboxes")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return react_var_inv, prod_var_inv

