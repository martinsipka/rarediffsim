import torch

BOLTZMAN = 0.001987191

def maxwell_boltzmann(masses, T, dimension, replicas=1):

    if masses.dim() == 2:
        rand_n = torch.randn((replicas,  dimension)).type_as(masses)
    elif masses.dim() == 3:
        rand_n = torch.randn((replicas,  masses.shape[1], 3)).type_as(masses)
    else:
        raise RuntimeError("Unknown number of dimensions")
    
    velocities = torch.sqrt(T * BOLTZMAN / masses) * rand_n
    return velocities

def kinetic_energy(masses, vel):
    Ekin = torch.sum(0.5 * torch.sum(vel * vel, dim=-1, keepdim=True) * masses, dim=1)
    return Ekin

def kinetic_to_temp(Ekin, dimension):
    return 2.0 / (dimension * BOLTZMAN) * Ekin
    
def temp_to_kin(T, dimension):
    return (dimension * BOLTZMAN) * T / 2.0 
    
    


