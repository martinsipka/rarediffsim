import torch


TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR

    
def simulateNVTSystem(system, force, args):
    
    T = args.temperature
    gamma = args.gamma / PICOSEC2TIMEU
    dt = args.dt / TIMEFACTOR

    M = system.M
    noise_f = torch.sqrt(2.0 * gamma / M * BOLTZMAN * T * dt).to(args.device)
    
    R = system.pos
    V = system.vel
    
    #dont preallocate the list to save positions, velocities and forces as it seemingly does not help. 
    pos_list = []
    vel_list = []
    fs_list = []
    
    f, f_b = force(R)
    tot_force = f+f_b
    V = V + dt/2*(tot_force/M - gamma*V) + noise_f/torch.sqrt(torch.tensor(2.0))*torch.randn_like(V)
    
    pos_list.append(R.clone())
    vel_list.append(V.clone())
    fs_list.append(tot_force)
    
    for i in range(1,args.iterations):

        R = R.detach()  + V*dt 
        #R = R + V*dt 
        #R.requires_grad=True
        
        f, f_b = force(R)
        tot_force =  f+f_b
        V = V + dt*(tot_force/M - gamma*V) + noise_f*torch.randn_like(V)
        
        if i % args.save_steps == 0:
            pos_list.append(R.clone())
            vel_list.append(V.clone())
            fs_list.append(tot_force)
 
    return pos_list, vel_list, fs_list[1:-1]
    

    
    
def simulateNVTSystem_adjoint(system, force, args):
    
    with torch.no_grad():
        T = args.temperature
        gamma = args.gamma / PICOSEC2TIMEU
        dt = args.dt / TIMEFACTOR

        M = system.M
        noise_f = torch.sqrt(2.0 * gamma / M * BOLTZMAN * T * dt).to(args.device)

        R = system.pos
        V = system.vel
        
        

        #dont preallocate the list to save positions, velocities and forces as it seemingly does not help. 
        pos_list = []
        vel_list = []

        tot_force = force(R)

        V = V + dt/2*(tot_force/M - gamma*V) + noise_f/torch.sqrt(torch.tensor(2.0))*torch.randn_like(V)


        pos_list.append(R.clone().cpu())
        vel_list.append(V.clone().cpu())
        
        for i in range(1,args.iterations):

            R += V*dt 
            
            tot_force = force(R)
            
            V = V + dt*(tot_force/M - gamma * V) + noise_f * torch.randn_like(V)

            if i % args.save_steps == 0:
                pos_list.append(R.clone().cpu())
                vel_list.append(V.clone().cpu())
     
    return pos_list, vel_list, None
    
def simulateNVTSystem_warmup(system, forces, args, steps=10):
    with torch.no_grad():
        T = args.temperature
        gamma = args.gamma / PICOSEC2TIMEU
        dt = args.dt / TIMEFACTOR

        M = system.M
        noise_f = torch.sqrt(2.0 * gamma / M * BOLTZMAN * T * dt).to(args.device)

        R = system.pos
        V = system.vel
        

        _, f_U = forces(R)
        V = V + dt/2*(f_U/M - gamma*V) + noise_f/torch.sqrt(torch.tensor(2.0))*torch.randn_like(V)

        pos_list = []
        
        for i in range(1,steps+1):

            R = R + V*dt 
            
            _, f_U = forces(R)
            
            V = V + dt*(f_U/M - gamma * V) + noise_f * torch.randn_like(V)
    
            if i % args.save_steps == 0:
                pos_list.append(R.clone().cpu())
    
    system.pos = R
    return pos_list

