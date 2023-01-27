import torch
import matplotlib.pyplot as plt

import numpy as np

const_pi = torch.tensor(np.pi)

class DescriptorLoss():

    def __init__(self, descriptors, react_var, prod_var, args, traj_loss=True):
        self.descriptors = descriptors
        self.loss_type = args.loss
        self.react_var = react_var.repeat(args.batch_size, 1,1)
        self.prod_var = prod_var.repeat(args.batch_size, 1,1)
        self.end_tol   = torch.cat((self.prod_var, self.react_var), axis=0)
        self.start_tol = torch.cat((self.react_var, self.prod_var), axis=0)
        self.traj_loss = traj_loss

        self.args = args 
        
    def maha_dist(self, d, var_mat):

        dim = d.shape[-1]
        summ = (d.unsqueeze(-2) @ var_mat @ d.unsqueeze(-1)).squeeze()
        return summ

    def two_point_distance(self, points, target, tol):
    
        if points.dim() > target.dim():
            target = target.unsqueeze(1)
            tol = tol.unsqueeze(1)

        d = self.descriptors.metric_from(points, target)
        
        if self.loss_type == "quad":
            dist = torch.sum(d**2,dim=-1)
        elif self.loss_type =="linear":
            dist = torch.sum(torch.abs(d),dim=-1)    

        use = self.maha_dist(d, tol) > self.args.domain_tol
        dist = use*dist

        return dist

    def distance_to_point(self, traj, target, tol): 

        dist = self.two_point_distance(traj, target, tol)
  
        dist, indices = torch.min(dist, dim=-1)
        closest_points = traj[torch.arange(traj.shape[0]), indices]
        
        return dist, indices
         
         
    #Given two sets (trajectories), return the indices of nearest neighbours and their distance. Respects batching. 
    def two_set_nearest_neighbour(self, traj, iterations=2):
        b_size = traj.shape[0]//2
        A, B = torch.split(traj, b_size, dim=0)
        targetB = B[:,-1]
        
        
        for i in range(iterations):
            #distances = torch.sum((A - targetB.unsqueeze(1))**2, dim=-1)
            distA, indicesA = self.distance_to_point(A, targetB.unsqueeze(1), self.prod_var.unsqueeze(1)/5)
            #distA, indicesA = self.distance_to_point(A, targetB.unsqueeze(1), torch.tensor([0]))
            targetA = A[torch.arange(b_size), indicesA]
            
            #distances = torch.sum((B - targetA.unsqueeze(1))**2, dim=-1)
            distB, indicesB = self.distance_to_point(B, targetA.unsqueeze(1), self.react_var.unsqueeze(1)/5)
            #distB, indicesB = self.distance_to_point(B, targetA.unsqueeze(1), torch.tensor([0]))
            targetB = B[torch.arange(b_size), indicesB]

        return torch.cat((distA,distB)), torch.cat((indicesA, indicesB)), torch.cat((targetA, targetB))
        
    #Constructed as follows   
    #Discard first third of the points. 
    #Calculated distance to the other trajectory, and to both basins. 
    #If under eps, point to the basin, if over eps point to other trajectory
    #The magnitude will be the distance from the basin
    #Distance to home is always included in the loss function
    #
    def dist_loss_adjoint(self, traj_tensor, target, start):
        
        with torch.no_grad():
            if not traj_tensor.requires_grad:
                traj_tensor.requires_grad=True
            start_offset = self.args.warmup
            start_traj_tensor = traj_tensor[:,start_offset:]

            dist_end, indices_end  = self.distance_to_point(traj_tensor, target, self.end_tol)
            loss_start, indices_start = self.distance_to_point(start_traj_tensor, start, self.start_tol)
        
        #loss_end = torch.sum(loss_end)
        #loss_start = torch.sum(loss_start)
        
        indices_start = indices_start + start_offset

        r_t = torch.arange(traj_tensor.shape[0])
        
        #Lets calculate again, this time with grad to have aN. First time we did without grad to save mem
        end_coordinate = traj_tensor[r_t, indices_end]
        if not end_coordinate.requires_grad:
            end_coordinate.requires_grad = True
        loss_end = self.two_point_distance(end_coordinate, target, self.end_tol)
        aN_end = torch.autograd.grad(torch.sum(loss_end), end_coordinate)[0].detach()
        

        
        start_coordinate = traj_tensor[r_t, indices_start]
        if not start_coordinate.requires_grad:
            start_coordinate.requires_grad = True
        loss_start = self.two_point_distance(start_coordinate, start, self.start_tol)
        aN_start = torch.autograd.grad(torch.sum(loss_start), start_coordinate)[0].detach()
        
        hit_prob = self.maha_dist(self.descriptors.metric_from(end_coordinate, target), self.end_tol)

        success_rate = torch.sum(hit_prob < self.args.domain_tol)/self.args.batch_size/2
        
        if self.traj_loss:
            dist_to_other, indices_trajs, targets_traj = self.two_set_nearest_neighbour(start_traj_tensor)
            traj_coordinate = traj_tensor[r_t, indices_trajs]
            if not traj_coordinate.requires_grad:
                traj_coordinate.requires_grad = True
            loss_traj = self.two_point_distance(traj_coordinate, targets_traj, self.end_tol)
            aN_trajs = torch.autograd.grad(torch.sum(loss_traj), traj_coordinate)[0].detach()
            
            loss_end = torch.sum(torch.where(dist_to_other > 1e-6, dist_to_other, dist_end))
            indices_end = torch.where(dist_to_other > 1e-6, indices_trajs, indices_end)
            aN_end = torch.where(dist_to_other.repeat(traj_tensor.shape[-1],1).transpose(0,1) > 1e-6, aN_trajs, aN_end)        

        return loss_end.sum(), loss_start.sum(), indices_end, aN_end, indices_start, aN_start, success_rate
        
