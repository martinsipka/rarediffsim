import torch
import numpy as np

#No need to wrap at this point. We can just get positions and 
#calculate dihedrals as the traj_tensor is not wrapped
#No need to wrap at this point. We can just get positions and 
#calculate dihedrals as the traj_tensor is not wrapped



class DihedralDescriptors():
    
    def __init__(self, indices=None, cv_index=None):
        self.ind = indices
        self.cv_index = cv_index

        
    #Get plain descriptors
    def get_descriptors(self, R):
        a0 = R[...,self.ind[0],:]
        a1 = R[...,self.ind[1],:]
        a2 = R[...,self.ind[2],:]
        a3 = R[...,self.ind[3],:]
        r12 = a1-a0
        r23 = a2-a1
        r34 = a3-a2
        crossA = torch.cross(r12, r23, dim=-1)
        crossB = torch.cross(r23, r34, dim=-1)
        crossC = torch.cross(r23, crossA, dim=-1)
        normA = torch.norm(crossA, dim=-1)
        normB = torch.norm(crossB, dim=-1)
        normC = torch.norm(crossC, dim=-1)
        normcrossB = crossB / normB.unsqueeze(-1)
        cosPhi = torch.sum(crossA * normcrossB, dim=-1) / normA
        sinPhi = torch.sum(crossC * normcrossB, dim=-1) / normC
        dihedrals = torch.atan2(sinPhi, cosPhi)
        return dihedrals
        
    #Get metric - How far is the descriptor from actual target. Account for periodicity etc. Distances are non-periodic. Also acount for rescaling. 
    #Periodicity from -pi to pi
    def metric_from(self, R, C, r_to_desc=True, c_to_desc=True):
        if r_to_desc:
            R = self.get_descriptors(R)
        if c_to_desc:
            C = self.get_descriptors(C)
        d_uncor = R - C
        d_part = torch.where(d_uncor > np.pi, d_uncor-2*np.pi, d_uncor)
        d = torch.where(d_part <= -np.pi, d_part+2*np.pi, d_part)
        return d
   
   
class DistanceDescriptors():
    
    def __init__(self, indices=None, cv_index=None):
        self.ind = indices
        self.cv_index = cv_index
    
    def get_descriptors(self, R):
        a0 = R[...,self.ind[0],:]
        a1 = R[...,self.ind[1],:]
        
        r12 = a1-a0
        d = torch.norm(r12, dim=-1)
        return d
        
    #Get metric - How far is the descriptor from actual target. Account for periodicity etc. Distances are non-periodic. Also acount for rescaling. 
    def metric_from(self, R, C, r_to_desc=True, c_to_desc=True):
        if r_to_desc:
            R = self.get_descriptors(R)
        if c_to_desc:
            C = self.get_descriptors(C)
        d = R - C
        return d
        
        
class CoordinateDescriptors():
    
    def __init__(self, indices=None, cv_index=None):
        self.ind = indices
        self.cv_index = cv_index
    
    def get_descriptors(self, R):
        
        if self.ind:
            return R[...,self.ind,:]
        
        return R
        
        
    def metric_from(self, R, C, r_to_desc=True, c_to_desc=True):
        if r_to_desc:
            R = self.get_descriptors(R)
        if c_to_desc:
            C = self.get_descriptors(C)
        d = R - C
        return d
        
        
        

